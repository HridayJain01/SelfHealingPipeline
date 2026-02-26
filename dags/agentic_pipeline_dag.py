import csv
import itertools
import json
import os
import re
from datetime import timedelta, datetime

from airflow.sdk import dag, task, Param, get_current_context
import logging

logger = logging.getLogger(__name__)

class Config:

    BASE_DIR = os.getenv('PIPELINE_BASE_DIR', '/Users/hridayjain/Documents/Projects/SelfHealingDataPipeline')
    INPUT_FILE = os.getenv('PIPELINE_INPUT_FILE',
                           f'{BASE_DIR}/input/yelp_academic_dataset_review.json')
    OUTPUT_DIR = os.getenv('PIPELINE_OUTPUT_DIR',
                           f'{BASE_DIR}/output/')

    MAX_TEXT_LENGTH = int(os.getenv('PIPELINE_MAX_TEXT_LENGTH', 2000))
    DEFAULT_BATCH_SIZE = 1000  # increased from 100 for larger dataset processing
    DEFAULT_OFFSET = 0

    # Field auto-detection candidates (checked in order)
    TEXT_FIELD_CANDIDATES  = ['text', 'review', 'content', 'body', 'comment', 'description', 'message', 'review_text']
    ID_FIELD_CANDIDATES    = ['review_id', 'id', 'record_id', 'doc_id', 'item_id']
    RATING_FIELD_CANDIDATES = ['stars', 'rating', 'score', 'grade', 'num_stars', 'star_rating']

    # Validation thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv('PIPELINE_CONFIDENCE_THRESHOLD', 0.60))
    STAR_AGREEMENT_HIGH  = 4   # stars >= this → expect POSITIVE
    STAR_AGREEMENT_LOW   = 2   # stars <= this → expect NEGATIVE

    #OLLAMA SETTINGS
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 120))
    OLLAMA_RETRIES = int(os.getenv('OLLAMA_RETRIES', 3))

default_args = {
    'owner': 'Hriday Jain',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
    'execution_timeout': timedelta(minutes=120),  # extended for larger batches
}

def _load_ollama_model(model_name: str):
    import ollama

    logger.info(f"Loading OLLAMA model: {model_name}")
    logger.info(f'OLLAMA Host: {Config.OLLAMA_HOST}')

    client = ollama.Client(host=Config.OLLAMA_HOST)

    try:
        client.show(model_name)
        logger.info(f'OLLAMA model {model_name} is available.')
    except ollama.ResponseError as e:
        logger.info('Model not found locally. Attempting to pull from remote repository...')
        try:
            client.pull(model_name)
            logger.info(f'OLLAMA model {model_name} pulled successfully.')
        except ollama.ResponseError as pull_error:
            logger.error(f'Failed to pull model {model_name}: {pull_error}')
            raise

    test_response = client.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": "Classify the sentiment: 'This is a great product!' as positive, negative, or neutral."
        }])
    test_result = test_response['message']['content'].strip().upper()
    logger.info(f'Model validation passed with test response: {test_result}')

    return {
        'backend': 'ollama',
        'model_name': model_name,
        'ollama_host': Config.OLLAMA_HOST,
        'max_length': Config.MAX_TEXT_LENGTH,
        'status': 'loaded',
        'validated_at': datetime.now().isoformat(),
    }

def _detect_fields(record: dict) -> dict:
    """Auto-detect which field names carry text, id, and rating in an arbitrary record."""
    keys = set(record.keys())

    text_field = next((f for f in Config.TEXT_FIELD_CANDIDATES if f in keys), None)
    if text_field is None:
        # fall back to any field whose value looks like a sentence
        text_field = next(
            (k for k, v in record.items() if isinstance(v, str) and len(v) > 20),
            list(keys)[0] if keys else 'text'
        )

    id_field = next((f for f in Config.ID_FIELD_CANDIDATES if f in keys), None)
    rating_field = next((f for f in Config.RATING_FIELD_CANDIDATES if f in keys), None)

    return {'text': text_field, 'id': id_field, 'rating': rating_field}


def _normalise_record(raw: dict, field_map: dict, override_text: str | None, override_id: str | None, override_rating: str | None) -> dict:
    """Map an arbitrary record to the internal schema used by the rest of the pipeline."""
    tf = override_text   or field_map['text']
    idf = override_id   or field_map['id']
    rf  = override_rating or field_map['rating']

    # Collect every key not mapped to core fields as metadata
    core = {tf, idf, rf} - {None}
    metadata = {k: v for k, v in raw.items() if k not in core}

    return {
        'review_id':   raw.get(idf)  if idf  else None,
        'business_id': raw.get('business_id'),
        'user_id':     raw.get('user_id'),
        'stars':       raw.get(rf, 0) if rf else 0,
        'text':        raw.get(tf),
        'date':        raw.get('date'),
        'useful':      raw.get('useful', 0),
        'funny':       raw.get('funny',  0),
        'cool':        raw.get('cool',   0),
        '_extra':      metadata,   # preserve all unknown fields
    }


def _load_from_file(params: dict, batch_size: int, offset: int):
    """Load records from JSONL, JSON-array, or CSV files with automatic field detection."""
    input_file    = params.get('input_file',    Config.INPUT_FILE)
    text_field    = params.get('text_field')    or None
    id_field      = params.get('id_field')      or None
    rating_field  = params.get('rating_field')  or None

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ext = os.path.splitext(input_file)[1].lower()
    records: list[dict] = []
    field_map: dict | None = None

    # ── CSV ──────────────────────────────────────────────────────────────
    if ext == '.csv':
        with open(input_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(itertools.islice(reader, offset, offset + batch_size)):
                raw = dict(row)
                if field_map is None:
                    field_map = _detect_fields(raw)
                    logger.info(f'CSV field mapping detected: {field_map}')
                records.append(_normalise_record(raw, field_map, text_field, id_field, rating_field))

    # ── JSON array or JSONL ───────────────────────────────────────────────
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                # Full JSON array – load entire file then slice
                try:
                    all_records = json.load(f)
                    sliced = itertools.islice(all_records, offset, offset + batch_size)
                    for raw in sliced:
                        if not isinstance(raw, dict):
                            continue
                        if field_map is None:
                            field_map = _detect_fields(raw)
                            logger.info(f'JSON-array field mapping detected: {field_map}')
                        records.append(_normalise_record(raw, field_map, text_field, id_field, rating_field))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON array from {input_file}: {e}") from e
            else:
                # JSONL – one JSON object per line
                sliced = itertools.islice(f, offset, offset + batch_size)
                for line_no, line in enumerate(sliced, start=offset + 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        if not isinstance(raw, dict):
                            logger.warning(f'Line {line_no}: expected dict, got {type(raw).__name__} – skipping')
                            continue
                        if field_map is None:
                            field_map = _detect_fields(raw)
                            logger.info(f'JSONL field mapping detected: {field_map}')
                        records.append(_normalise_record(raw, field_map, text_field, id_field, rating_field))
                    except json.JSONDecodeError as e:
                        logger.warning(f'Skipping invalid JSON at line {line_no}: {e}')
                        continue

    logger.info(
        f'Loaded {len(records)} records from "{os.path.basename(input_file)}" '
        f'(offset={offset}, batch_size={batch_size}, format={ext or "jsonl"})'
    )
    return records

def _parse_ollama_response(response_text: str):
    try:

        clean_text = response_text.strip()

        if clean_text.startswith('```'):
            lines = clean_text.split('\n')
            clean_text = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])

        parsed = json.loads(clean_text)
        sentiment = parsed.get('sentiment', 'NEUTRAL').upper()
        confidence = float(parsed.get('confidence', 0.0))

        if sentiment not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sentiment = 'NEUTRAL'

        return {
            'label': sentiment,
            'score': min(max(confidence, 0.0), 1.0)
        }
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        upper_text = response_text.strip().upper()
        if 'POSITIVE' in upper_text:
            return { 'label': 'POSITIVE', 'score': 0.75 }
        elif 'NEGATIVE' in upper_text:
            return { 'label': 'NEGATIVE', 'score': 0.75 }
        return { 'label': 'NEUTRAL', 'score': 0.5 }

def _heal_review(review: dict) -> dict:
    text = review.get('text', '')

    result = {
        'review_id': review.get('review_id'),
        'business_id': review.get('business_id'),
        'stars': review.get('stars', 0),
        'original_text': None,
        'error_type': None,
        'action_taken': 'none',
        'was_healed': False,
        'metadata': {
            'user_id': review.get('user_id'),
            'date': review.get('date'),
            'useful': review.get('useful', 0),
            'funny': review.get('funny', 0),
            'cool': review.get('cool', 0),
        }
    }

    if isinstance(text, (str, int, float, bool, type(None))):
        result['original_text'] = text
    else:
        result['original_text'] = str(text) if text else None

    if text is None:
        result['error_type'] = 'missing_text'
        result['action_taken'] = 'filled_with_placeholder'
        result['healed_text'] = 'No review text provided.'
        result['was_healed'] = True
        return result
    elif not isinstance(text, str):
        result['error_type'] = 'wrong_type'
        try:
            converted = str(text).strip()
            result['healed_text'] = converted if converted else 'No review text provided.'
        except Exception as e:
            result['healed_text'] = 'Conversion failed.'

        result['action_taken'] = 'type_conversion'
        result['was_healed'] = True
    elif not text.strip():
        result['error_type'] = 'empty_text'
        result['healed_text'] = 'No review text provided.'
        result['action_taken'] = 'filled_with_placeholder'
        result['was_healed'] = True
    elif not re.search(r'[a-zA-Z0-9]', text):
        result['error_type'] = 'special_characters_only'
        result['healed_text'] = '[Non-text content]'
        result['action_taken'] = 'replaced_special_characters'
        result['was_healed'] = True
    elif len(text) > Config.MAX_TEXT_LENGTH:
        result['error_type'] = 'too_long'
        result['healed_text'] = text[:Config.MAX_TEXT_LENGTH-3] + '...'
        result['action_taken'] = 'truncated_text'
        result['was_healed'] = True
    else:
        result['healed_text'] = text.strip()
        result['was_healed'] = False

    return result

def _analyze_with_ollama(healed_reviews: list[dict], model_info: dict) -> list[dict]:
    import ollama
    import time

    model_name = model_info.get('model_name')
    ollama_host = model_info.get('ollama_host', Config.OLLAMA_HOST)

    try:
        client = ollama.Client(host=ollama_host)
    except Exception as e:
        logger.error(f'Failed to connect to OLLAMA host {ollama_host}: {e}')
        return _created_degraded_results(healed_reviews, str(e))

    results = []
    total = len(healed_reviews)

    for idx, review in enumerate(healed_reviews):
        text = review.get('healed_text', '')
        prediction = None

        for attempt in range(Config.OLLAMA_RETRIES):
            try:
                prompt = f"""
                    Analyze the sentiment of this review and classify it as POSITIVE, NEGATIVE, or NEUTRAL. 
                    Review: "{text}"
                    Reply with ONLY a JSON object: {{"sentiment": "POSITIVE", "confidence": 0.95}}.
                    """

                response = client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={'temperature': 0.1},
                )

                response_text = response['message']['content'].strip()
                prediction = _parse_ollama_response(response_text)
                break

            except Exception as e:
                if attempt < Config.OLLAMA_RETRIES - 1:
                    logger.warning(f'Attempt {attempt+1} failed for review {review.get("review_id")}: {e}. Retrying...')
                    time.sleep(1)
                else:
                    logger.error(f'All attempts failed for review {review.get("review_id")}: {e}.')
                    prediction = {'label': 'NEUTRAL', 'score': 0.5, 'error': str(e)}

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            logger.info(f'Processed {idx + 1}/{total} reviews for sentiment analysis.')

        results.append({
            'review_id': review.get('review_id'),
            'business_id': review.get('business_id'),
            'stars': review.get('stars', 0),
            'text': review.get('healed_text', ''),
            'original_text': review.get('original_text', ''),
            'predicted_sentiment': prediction.get('label'),
            'confidence': round(prediction.get('score'), 4),
            'status': 'healed' if review.get('was_healed') else 'success',
            'healing_applied': review.get('was_healed'),
            'healing_action': review.get('action_taken') if review.get('was_healed') else None,
            'error_type': review.get('error_type') if review.get('was_healed') else None,
            'metadata': review.get('metadata', {}),
        })

    logger.info(f'Ollama inference complete: {len(results)}/{total} reviews processed.')
    return results

def _validate_results(analyzed_results: list[dict]) -> dict:
    """Post-inference validation: confidence gating and star-sentiment agreement."""
    total = len(analyzed_results)
    low_confidence: list[str] = []
    disagreements: list[dict] = []
    agreement_count = 0
    eligible_for_agreement = 0

    for r in analyzed_results:
        rid        = r.get('review_id', 'unknown')
        confidence = r.get('confidence', 0.0)
        sentiment  = r.get('predicted_sentiment', 'NEUTRAL')
        stars      = r.get('stars', 0)

        # Confidence gate
        if confidence < Config.CONFIDENCE_THRESHOLD:
            low_confidence.append(rid)

        # Star–sentiment agreement (only for records that have a numeric star rating)
        try:
            stars_val = float(stars)
        except (TypeError, ValueError):
            stars_val = None

        if stars_val is not None and stars_val > 0:
            eligible_for_agreement += 1
            if stars_val >= Config.STAR_AGREEMENT_HIGH:
                expected = 'POSITIVE'
            elif stars_val <= Config.STAR_AGREEMENT_LOW:
                expected = 'NEGATIVE'
            else:
                expected = None  # neutral zone – skip agreement check

            if expected is not None:
                if sentiment == expected:
                    agreement_count += 1
                else:
                    disagreements.append({
                        'review_id': rid,
                        'stars': stars_val,
                        'predicted': sentiment,
                        'expected': expected,
                        'confidence': confidence,
                    })

    agreement_rate = round(agreement_count / max(eligible_for_agreement, 1), 4)
    low_conf_rate  = round(len(low_confidence) / max(total, 1), 4)

    logger.info(
        f'Validation complete: {total} records | '
        f'agreement_rate={agreement_rate} ({agreement_count}/{eligible_for_agreement}) | '
        f'low_confidence={len(low_confidence)} ({low_conf_rate*100:.1f}%) | '
        f'disagreements={len(disagreements)}'
    )

    # Log top disagreements for visibility
    if disagreements:
        sample = disagreements[:10]
        logger.warning(f'Top star-sentiment disagreements (showing up to 10): {json.dumps(sample, default=str)}')

    return {
        'total_validated': total,
        'low_confidence_count': len(low_confidence),
        'low_confidence_rate': low_conf_rate,
        'low_confidence_threshold': Config.CONFIDENCE_THRESHOLD,
        'eligible_for_star_agreement': eligible_for_agreement,
        'agreement_count': agreement_count,
        'disagreement_count': len(disagreements),
        'agreement_rate': agreement_rate,
        'top_disagreements': disagreements[:20],
    }


def _created_degraded_results(healed_reviews: list[dict], error_message: str) -> list[dict]:
    return [
        {
            **review,
            'text': review.get('healed_text', ''),
            'predicted_sentiment': 'NEUTRAL',
            'confidence': 0.5,
            'status': 'degraded',
            'error_message': error_message,
        }
        for review in healed_reviews
    ]

@dag(
    dag_id='self_healing_pipeline',
    default_args=default_args,
    description="Pipeline for sentiment analysis using OLLAMA model",
    schedule=None,
    start_date=datetime(2025, 12, 7),
    catchup=False,
    tags=['sentiment_analysis', 'nlp', 'ollama', 'universal_input'],
    params={
        'input_file': Param(
            default=Config.INPUT_FILE,
            type='string',
            description='Path to the input file (JSONL, JSON array, or CSV).'
        ),
        'text_field': Param(
            default='',
            type='string',
            description='Field name containing the text to analyse. Leave blank for auto-detection.'
        ),
        'id_field': Param(
            default='',
            type='string',
            description='Field name for the record ID. Leave blank for auto-detection.'
        ),
        'rating_field': Param(
            default='',
            type='string',
            description='Field name for the numeric rating / stars. Leave blank for auto-detection.'
        ),
        'batch_size': Param(
            default=Config.DEFAULT_BATCH_SIZE,
            type='integer',
            description='Number of records to process per run (default 1000).'
        ),
        'offset': Param(
            default=Config.DEFAULT_OFFSET,
            type='integer',
            description='Record offset to start reading from.'
        ),
        'ollama_model': Param(
            default=Config.OLLAMA_MODEL,
            type='string',
            description='Name of the OLLAMA model to use for sentiment analysis.'
        ),
    },
    render_template_as_native_obj=True,
)
def self_healing_pipeline():
    @task
    def load_model():
        context = get_current_context()
        params = context['params']
        model_name = params.get('ollama_model', Config.OLLAMA_MODEL)
        logger.info(f'Using OLLAMA model: {model_name}')
        return _load_ollama_model(model_name)

    @task
    def load_reviews():
        context = get_current_context()
        params = context['params']
        batch_size = params.get('batch_size', Config.DEFAULT_BATCH_SIZE)
        offset = params.get('offset', Config.DEFAULT_OFFSET)
        logger.info(f'Loading reviews with batch size {batch_size} and offset {offset}')
        return _load_from_file(params, batch_size, offset)

    @task
    def diagnose_and_heal_batch(reviews: list[dict]):
        healed_reviews = [_heal_review(review) for review in reviews]
        healed_count = sum(1 for r in healed_reviews if r.get('was_healed', True))
        logger.info(f'Healed {healed_count} out of {len(reviews)} reviews in the batch.')
        return healed_reviews

    @task
    def batch_analyze_sentiment(healed_reviews: list[dict], model_info: dict):
        if not healed_reviews:
            return []
        logger.info(f'Analyzing {len(healed_reviews)} reviews for sentiment.')
        return _analyze_with_ollama(healed_reviews, model_info)

    @task
    def validate_results(analyzed_results: list[dict]) -> dict:
        """Confidence gating + star-sentiment agreement validation."""
        if not analyzed_results:
            logger.warning('validate_results received an empty list.')
            return {
                'total_validated': 0,
                'low_confidence_count': 0, 'low_confidence_rate': 0.0,
                'low_confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                'eligible_for_star_agreement': 0,
                'agreement_count': 0, 'disagreement_count': 0,
                'agreement_rate': 0.0, 'top_disagreements': [],
            }
        return _validate_results(analyzed_results)

    @task
    def aggregate_results(results: list[list[dict]], validation: dict):
        context = get_current_context()
        params = context['params']
        results = list(results)

        total = len(results)

        success_count = sum(1 for r in results if r.get('status') == 'success')
        healed_count = sum(1 for r in results if r.get('status') == 'healed')
        degraded_count = sum(1 for r in results if r.get('status') == 'degraded')

        sentiment_dist = { 'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0 }
        for r in results:
            sentiment = r.get('predicted_sentiment', 'NEUTRAL')
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1

        healing_stats = {}
        for r in results:
            if r.get('healing_applied'):
                action = r.get('healing_action', 'unknown')
                healing_stats[action] = healing_stats.get(action, 0) + 1

        star_sentiment = {}
        for r in results:
            stars = r.get('stars', 0)
            sentiment = r.get('predicted_sentiment')
            if stars and sentiment:
                key = f'{int(stars)}_star'
                if key not in star_sentiment:
                    star_sentiment[stars] = { 'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0 }
                star_sentiment[stars][sentiment] += 1

        confidence_by_status = { 'success': [], 'healed': [], 'degraded': [] }
        for r in results:
            status = r.get('status')
            confidence = r.get('confidence', 0.0)
            if status in confidence_by_status:
                confidence_by_status[status].append(confidence)

        avg_confidence = {
            status: (sum(conf_list) / len(conf_list)) if conf_list else 0
            for status, conf_list in confidence_by_status.items()
        }

        summary = {
            "run_info": {
                'timestamp': datetime.now().isoformat(),
                'batch_size': params.get('batch_size', Config.DEFAULT_BATCH_SIZE),
                'offset': params.get('offset', Config.DEFAULT_OFFSET),
                'input_file': params.get('input_file', Config.INPUT_FILE),
                'text_field':   params.get('text_field')   or '(auto)',
                'id_field':     params.get('id_field')     or '(auto)',
                'rating_field': params.get('rating_field') or '(auto)',
            },
            'totals': {
                'processed': total,
                'success': success_count,
                'healed': healed_count,
                'degraded': degraded_count,
            },
            'rates': {
                'success_rate': round(success_count / max(total, 1), 4),
                'healing_rate': round(healed_count / max(total, 1), 4),
                'degradation_rate': round(degraded_count / max(total, 1), 4),
            },
            'sentiment_distribution': sentiment_dist,
            'healing_statistics': healing_stats,
            'star_sentiment_correlation': star_sentiment,
            'average_confidence': avg_confidence,
            'validation': validation,
            'results': results,
        }

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        offset = params.get('offset', Config.DEFAULT_OFFSET)
        output_file = f'{Config.OUTPUT_DIR}/sentiment_analysis_summary_{timestamp}_Offset{offset}.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f'Summary written to {output_file}')
        logger.info(f'Processed {total} reviews: {success_count} success, {healed_count} healed, {degraded_count} degraded.')

        return { k: v for k, v in summary.items() if k != 'results' }

    @task
    def generate_health_report(summary: dict):
        total    = summary['totals']['processed']
        healed   = summary['totals']['healed']
        degraded = summary['totals']['degraded']
        val       = summary.get('validation', {})
        agree_rate = val.get('agreement_rate', 1.0)
        low_conf_rate = val.get('low_confidence_rate', 0.0)
        healing_rate  = summary['rates']['healing_rate']
        degradation_rate = summary['rates']['degradation_rate']

        # Multi-factor health scoring
        issues = []
        if degradation_rate > 0.10:
            issues.append(f'HIGH degradation ({degradation_rate*100:.1f}%)')
        if low_conf_rate > 0.30:
            issues.append(f'HIGH low-confidence rate ({low_conf_rate*100:.1f}% < threshold {Config.CONFIDENCE_THRESHOLD})')
        if agree_rate < 0.60 and val.get('eligible_for_star_agreement', 0) > 10:
            issues.append(f'LOW star-sentiment agreement ({agree_rate*100:.1f}%)')
        if healing_rate > 0.20:
            issues.append(f'HIGH healing rate ({healing_rate*100:.1f}%) – data quality concern')

        if degradation_rate > 0.10 or (agree_rate < 0.50 and val.get('eligible_for_star_agreement', 0) > 10):
            health_status = 'CRITICAL'
        elif issues:
            health_status = 'WARNING'
        elif healing_rate > 0.05:
            health_status = 'DEGRADED'  # pipeline ran but had to heal > 5% of records
        else:
            health_status = 'HEALTHY'

        report = {
            'pipeline': 'self_healing_pipeline',
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'issues_detected': issues,
            'run_info': summary['run_info'],
            'metrics': {
                'total_processed': total,
                'success_rate': summary['rates']['success_rate'],
                'healing_rate': healing_rate,
                'degradation_rate': degradation_rate,
            },
            'sentiment_distribution': summary['sentiment_distribution'],
            'healing_summary': summary['healing_statistics'],
            'average_confidence': summary['average_confidence'],
            'validation_metrics': {
                'agreement_rate': agree_rate,
                'disagreement_count': val.get('disagreement_count', 0),
                'low_confidence_count': val.get('low_confidence_count', 0),
                'low_confidence_rate': low_conf_rate,
                'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
            },
        }

        logger.info(f'Pipeline Health Report: {json.dumps(report, indent=2)}')
        logger.info(
            f'Status={health_status} | '
            f'success={summary["rates"]["success_rate"]:.2%} '
            f'healed={healing_rate:.2%} '
            f'degraded={degradation_rate:.2%} '
            f'agreement={agree_rate:.2%} '
            f'low_conf={low_conf_rate:.2%}'
        )
        if issues:
            logger.warning(f'Issues detected: {"; ".join(issues)}')

        return report

    model_info = load_model()
    reviews = load_reviews()

    healed_reviews = diagnose_and_heal_batch(reviews)
    analyzed_results = batch_analyze_sentiment(healed_reviews, model_info)

    validation = validate_results(analyzed_results)
    summary = aggregate_results(analyzed_results, validation)
    health_report = generate_health_report(summary)

self_healing_pipeline_dag = self_healing_pipeline()