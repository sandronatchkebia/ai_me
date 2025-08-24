#!/usr/bin/env python3
"""
New pre-processing script with proper three-bucket classification:
1. Georgian (Unicode): [\u10A0-\u10FF] - 100% reliable
2. Georgian (English Keyboard): Romanized Georgian using heuristics
3. English: Clean English text

Don't rely on existing language classification - reclassify everything.
"""

import argparse, json, re, hashlib, unicodedata, random
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

# Georgian Unicode range
GEORGIAN_UNICODE_RANGE = re.compile(r'[\u10A0-\u10FF]')

# English dictionary (common words)
ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    'hello', 'hi', 'thanks', 'thank', 'you', 'yes', 'no', 'ok', 'okay', 'good', 'bad', 'great', 'nice', 'cool', 'awesome', 'amazing', 'wow', 'oh', 'hey', 'what',
    'how', 'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those', 'here', 'there', 'now', 'then', 'today', 'tomorrow', 'yesterday', 'week', 'month', 'year',
    'work', 'home', 'school', 'university', 'college', 'office', 'meeting', 'call', 'email', 'message', 'text', 'phone', 'computer', 'internet', 'website', 'app', 'program',
    'food', 'drink', 'water', 'coffee', 'tea', 'breakfast', 'lunch', 'dinner', 'snack', 'restaurant', 'cafe', 'bar', 'hotel', 'travel', 'trip', 'vacation', 'holiday',
    'family', 'friend', 'people', 'person', 'man', 'woman', 'boy', 'girl', 'child', 'baby', 'parent', 'mother', 'father', 'sister', 'brother', 'son', 'daughter',
    'love', 'like', 'hate', 'feel', 'think', 'know', 'understand', 'believe', 'hope', 'want', 'need', 'must', 'should', 'can', 'will', 'would', 'could', 'might',
    'big', 'small', 'large', 'tiny', 'huge', 'enormous', 'short', 'long', 'tall', 'wide', 'narrow', 'high', 'low', 'deep', 'shallow', 'heavy', 'light', 'strong', 'weak',
    'fast', 'slow', 'quick', 'rapid', 'gradual', 'sudden', 'immediate', 'delayed', 'early', 'late', 'on', 'time', 'punctual', 'tardy', 'recent', 'old', 'new', 'fresh', 'stale',
    'hot', 'cold', 'warm', 'cool', 'freezing', 'boiling', 'sunny', 'cloudy', 'rainy', 'snowy', 'windy', 'calm', 'stormy', 'clear', 'foggy', 'humid', 'dry', 'wet', 'damp',
    # Additional common words
    'apologies', 'delayed', 'response', 'competition', 'looks', 'extremely', 'interesting', 'sure', 'exception', 'technical', 'skills', 'pull', 'off', 'limit', 'number', 'graduate', 'students', 'participate', 'best', 'regards', 'aleks', 'aleksandre', 'raja', 'ran', 'jiang', 'email', 'link', 'phone', 'number', 'future', 'worries', 'save', 'absolutely', 'make', 'person', 'zoom', 'davis', 'hall', 'pm', 'extremely', 'helpful', 'time', 'meet', 'friends', 'following', 'up', 'take', 'look', 'interested', 'apologies', 'delayed', 'response', 'competition', 'looks', 'extremely', 'interesting', 'sure', 'exception', 'technical', 'skills', 'pull', 'off', 'limit', 'number', 'graduate', 'students', 'participate', 'best', 'regards'
}

# Georgian stopwords in Latin script (only uniquely Georgian words)
GEORGIAN_STOPWORDS = {
        # Basic words
        'da', 'rogor', 'kargad', 'chemi', 'sheni', 'zalian', 'unda', 'rodesac', 'magram', 'imito', 'amis',
        'ara', 'aris', 'var', 'xar', 'xart', 'vart', 'gamarjoba', 'nakhvamdis', 'madloba', 'arafris',
        'shen', 'tkven', 'chven', 'saxli', 'mama', 'deda', 'dzma', 'bavshvi', 'gogo', 'bichi',
        'ra', 'vin', 'sad', 'rodis', 'ratom', 'rogor', 'xo', 'aris', 'ra aris', 'vin aris',
        'gamarjoba', 'rogor', 'xar', 'madloba', 'dzalian', 'chemi', 'sheni', 'saxli', 'dzma',
        
        # Additional common Georgian words
        'tu', 'rodesac', 'sad', 'ratom', 'rogor', 'vin', 'ra', 'sadac', 'rodisac', 'ratomac',
        'gvaketebs', 'gvaketeb', 'gvaketebt', 'gvaketebi', 'gvaketebs', 'gvaketebt',
        'vaketebs', 'vaketeb', 'vaketebt', 'vaketebi', 'vaketebs', 'vaketebt',
        'aketebs', 'aketeb', 'aketebt', 'aketebi', 'aketebs', 'aketebt',
        'ketebs', 'keteb', 'ketebt', 'ketebi', 'ketebs', 'ketebt',
        'etebs', 'eteb', 'etebt', 'etebi', 'etebs', 'etebt',
        'tebs', 'teb', 'tebt', 'tebi', 'tebs', 'tebt',
        'ebs', 'eb', 'ebt', 'ebi', 'ebs', 'ebt',
        'bs', 'b', 'bt', 'bi', 'bs', 'bt',
        
        # Common Georgian verbs and forms (removed single letters that are common in English)
        'vici', 'gici', 'cici', 'ci',
        'vxedav', 'gxedav', 'xedav', 'edav', 'dav', 'av',
        'vaketeb', 'gaketeb', 'aketeb', 'keteb', 'eteb', 'teb', 'eb',
        'vaketebs', 'gaketebs', 'aketebs', 'ketebs', 'etebs', 'tebs', 'ebs',
        
        # Georgian question words and particles
        'tu', 'ara', 'araa', 'ar aris', 'ar var', 'ar xar', 'ar xart', 'ar vart',
        'sad', 'rodis', 'ratom', 'rogor', 'vin', 'ra', 'sadac', 'rodisac', 'ratomac',
        
        # Georgian connectors and modifiers
        'da', 'magram', 'imito', 'amis', 'imis', 'amas', 'imas', 'amas', 'imas',
        'rodesac', 'sadac', 'rodisac', 'ratomac', 'vinmac', 'ramac', 'sadmac',
        
        # Georgian time and place words
        'axla', 'dges', 'ghvini', 'kvin', 'kvinad', 'kvinadac', 'kvinadacac',
        'sad', 'sadac', 'sadacac', 'sadacacac', 'sadacacacac',
        'rodis', 'rodisac', 'rodisacac', 'rodisacacac', 'rodisacacacac',
        
        # Georgian emotional and descriptive words
        'zalian', 'dzalian', 'dzalianad', 'dzalianadac', 'dzalianadacac',
        'kargad', 'kargadac', 'kargadacac', 'kargadacacac',
        'ubralod', 'ubralodac', 'ubralodacac', 'ubralodacacac',
        'martla', 'martlaac', 'martlaacac', 'martlaacacac',
        
        # Georgian common phrases
        'gamarjoba', 'nakhvamdis', 'madloba', 'arafris', 'arafrisac',
        'shen', 'tkven', 'chven', 'saxli', 'mama', 'deda', 'dzma', 'bavshvi', 'gogo', 'bichi',
        'gamarjoba', 'rogor', 'xar', 'madloba', 'dzalian', 'chemi', 'sheni', 'saxli', 'dzma'
    }

# Georgian character clusters
GEORGIAN_CLUSTERS = {
    # Basic clusters
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    
    # Extended clusters
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    
    # Common Georgian letter combinations
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    
    # Additional patterns
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    
    # Very specific Georgian patterns
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    
    # Common in Georgian names and words
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    
    # Additional Georgian patterns
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q',
    'kh', 'gh', 'ts', 'dz', 'sh', 'ch', 'zh', 'q'
}

# Georgian word endings
GEORGIAN_ENDINGS = {'ad', 'it', 'ul', 'ia', 'uri', 'ena', 'ebi', 'ebs', 'eb', 'ebt', 'eli', 'uli', 'ani', 'eni', 'oni', 'uni'}

# Noise text detection patterns
FORWARD_HDR = re.compile(r'(?i)^[-\s>]*forwarded message', re.M)
HEADER_BLOCK = re.compile(r'(?i)^(from|to|subject|date):', re.M)
RECEIPT_CUES = re.compile(
    r'(?i)\b(order (no\.?|number)|receipt|invoice|total|subtotal|customer number|'
    r'delivery method|unsubscribe|no-?reply|verification code|otp)\b'
)
MANY_NUMS = re.compile(r'\d')

MEDIA_PLACEHOLDER = re.compile(r'(?i)\b(image omitted|video omitted|audio omitted|sticker omitted|gif omitted|\[link\])\b')

# Email cleaning patterns
EMAIL_QUOTE_PATTERNS = [
    r"(?m)^\s*On\s.+?wrote:\s*$",  # "On ... wrote:"
    r"(?m)^\s*From:\s+.+?\s+Sent:\s+.+?\s+To:\s+.+?\s+Subject:\s+.+?$",  # Email headers
    r"(?m)^\s*From:\s+.+?\s+Date:\s+.+?\s+To:\s+.+?\s+Subject:\s+.+?$",  # Alternative header format
    r"(?m)^\s*Sent from my iPhone\s*$",  # Mobile signatures
    r"(?m)^\s*Sent from my iPad\s*$",  # iPad signatures
    r"(?m)^\s*Get Outlook for iOS\s*$",  # Outlook mobile
    r"(?m)^\s*CONFIDENTIALITY CAUTION AND DISCLAIMER.*$",  # Legal disclaimers
    r"(?m)^\s*This message is intended only for.*$",  # Legal disclaimers
    r"(?m)^\s*All unintended recipients are obliged.*$",  # Legal disclaimers
    r"(?m)^\s*epam\.com\s*$",  # Company URLs
    r"(?m)^\s*Mobile:\s+\+?\d+\s*$",  # Phone numbers
    r"(?m)^\s*Email:\s+[^\s]+\s*$",  # Email addresses
    r"(?m)^\s*Minsk, Belarus\s*$",  # Location info
    r"(?m)^\s*Best regards,\s*$",  # Common closings
    r"(?m)^\s*Thanks,\s*$",  # Common closings
    r"(?m)^\s*Thank you,\s*$",  # Common closings
    r"(?m)^\s*Sincerely,\s*$",  # Common closings
    r"(?m)^\s*Regards,\s*$",  # Common closings
    r"(?m)^\s*--\s*[ა-ჰ]+\s*$",  # Georgian signatures like "-- [Name]"
    r"(?m)^\s*--\s*[A-Za-z]+\s*[A-Za-z]*\s*$",  # English signatures like "-- [Name]"
    r"(?m)^\s*Best,\s*[A-Za-z]+\s*$",  # "Best, [Name]"
    r"(?m)^\s*Thanks,\s*[A-Za-z]+\s*$",  # "Thanks, [Name]"
    r"(?m)^\s*Thank you,\s*[A-Za-z]+\s*$",  # "Thank you, [Name]"
    r"(?m)^\s*Best regards,\s*[A-Za-z]+\s*$",  # "Best regards, [Name]"
    r"(?m)^\s*Regards,\s*[A-Za-z]+\s*$",  # "Regards, [Name]"
    r"(?m)^\s*Sincerely,\s*[A-Za-z]+\s*$",  # "Sincerely, [Name]"
    r"(?m)^\s*Cheers,\s*[A-Za-z]+\s*$",  # "Cheers, [Name]"
    r"(?m)^\s*Yours,\s*[A-Za-z]+\s*$",  # "Yours, [Name]"
    r"(?m)^\s*Kind regards,\s*[A-Za-z]+\s*$",  # "Kind regards, [Name]"
    r"(?m)^\s*Warm regards,\s*[A-Za-z]+\s*$",  # "Warm regards, [Name]"
    r"(?m)^\s*Take care,\s*[A-Za-z]+\s*$",  # "Take care, [Name]"
    r"(?m)^\s*All the best,\s*[A-Za-z]+\s*$",  # "All the best, [Name]"
    r"(?m)^\s*Best wishes,\s*[A-Za-z]+\s*$",  # "Best wishes, [Name]"
]

# Tech artifacts
TECH_ARTIFACTS = [
    r'<x-apple-data-detectors://\d+>',  # Apple data detectors
    r'<mailto:[^>]+>',  # Mailto links
    r'\[EMAIL\]',  # Redacted email placeholders
    r'\[LINK\]',  # Redacted link placeholders
    r'\[PHONE\]',  # Redacted phone placeholders
]

def classify_text(text):
    """
    Classify text into three buckets:
    1. Georgian (Unicode): Contains Georgian characters
    2. Georgian (English Keyboard): Romanized Georgian
    3. English: Clean English
    
    Improved logic to better distinguish English vs Georgian Latin script
    """
    if not text or len(text.strip()) < 2:
        return "unknown"
    
    text = text.strip()
    
    # 0) Easy win: Georgian Unicode characters
    if GEORGIAN_UNICODE_RANGE.search(text):
        return "ka"
    
    # 1) Check if ASCII-only
    if not text.isascii():
        return "unknown"  # Contains non-ASCII but not Georgian
    
    # 2) ASCII-only: Decide English vs Romanized Georgian
    text_lower = text.lower()
    words = text_lower.split()
    
    # Handle very short texts first
    if len(words) == 1:
        word = words[0]
        if word in GEORGIAN_STOPWORDS:
            return "ka_en"
        elif word in {'ok', 'yes', 'no', 'hi', 'hey', 'bye', 'cool', 'nice', 'good', 'bad', 'wow', 'oh', 'ah', 'um', 'uh'}:
            return "en"
        else:
            # default to English for single unknown words to avoid dropping
            return "en"
    elif len(words) < 2:
        # default to English rather than unknown to maximize retention
        return "en"
    
    # STRICT ENGLISH DETECTION - Must pass multiple criteria
    
    # Criterion 1: High English dictionary hit-rate (increased threshold)
    english_word_count = sum(1 for word in words if word in ENGLISH_WORDS)
    english_ratio = english_word_count / len(words)
    
    # Criterion 2: English pattern matches (increased threshold)
    english_patterns = [
        r'\b(hi|hello|hey|good|morning|afternoon|evening|night|day|week|month|year)\b',
        r'\b(thanks|thank|you|please|sorry|excuse|me|yes|no|ok|okay|sure|fine|great|good|bad|nice|cool|awesome|amazing|wow|oh|hey|what|how|when|where|why|who|which|that|this|these|those|here|there|now|then|today|tomorrow|yesterday)\b',
        r'\b(work|home|school|university|college|office|meeting|call|email|message|text|phone|computer|internet|website|app|program|food|drink|water|coffee|tea|breakfast|lunch|dinner|snack|restaurant|cafe|bar|hotel|travel|trip|vacation|holiday|family|friend|people|person|man|woman|boy|girl|child|baby|parent|mother|father|sister|brother|son|daughter)\b'
    ]
    
    pattern_matches = 0
    for pattern in english_patterns:
        matches = re.findall(pattern, text_lower)
        pattern_matches += len(matches)
    
    # Criterion 3: Check for English sentence structure (articles, prepositions, common verbs)
    english_structure_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
    structure_word_count = sum(1 for word in words if word in english_structure_words)
    structure_ratio = structure_word_count / len(words)
    
    # ENGLISH CLASSIFICATION: More lenient - pass 2 out of 3 criteria
    english_score = 0
    if english_ratio >= 0.6:  # Lowered from 0.7
        english_score += 1
    if pattern_matches >= len(words) * 0.3:  # Lowered from 0.5
        english_score += 1
    if structure_ratio >= 0.2:  # Lowered from 0.3
        english_score += 1
    
    # Classify as English if it passes 2 out of 3 criteria
    if english_score >= 2:
        return "en"
    
    # GEORGIAN LATIN SCRIPT DETECTION - More specific signals
    
    # Signal 1: Georgian character clusters (increased threshold and more specific)
    # Only count clusters that are NOT common in English words
    georgian_cluster_count = 0
    for word in words:
        # Skip very common English words that might contain these clusters
        if word in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'want', 'way', 'year', 'good', 'look', 'take', 'time', 'work', 'back', 'come', 'give', 'know', 'life', 'make', 'most', 'over', 'some', 'than', 'them', 'very', 'when', 'will', 'with', 'your', 'about', 'after', 'again', 'could', 'first', 'great', 'other', 'should', 'think', 'through', 'water', 'where', 'which', 'world', 'would', 'always', 'because', 'between', 'different', 'everything', 'something', 'sometimes', 'together', 'without', 'questions', 'should', 'reschedule', 'wants', 'poor', 'remain', 'rich', 'richer', 'michael', 'socialist'}:
            continue
            
        for cluster in GEORGIAN_CLUSTERS:
            if cluster in word:
                georgian_cluster_count += 1
                break
    
    cluster_ratio = georgian_cluster_count / len(words)
    if cluster_ratio >= 0.6:  # Increased from 0.5
        return "ka_en"
    
    # Signal 2: Georgian word endings (increased threshold and more specific)
    # Only count endings that are NOT common in English words
    ending_count = 0
    for word in words:
        # Skip very common English words that might have these endings
        if word in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'want', 'way', 'year', 'good', 'look', 'take', 'time', 'work', 'back', 'come', 'give', 'know', 'life', 'make', 'most', 'over', 'some', 'than', 'them', 'very', 'when', 'will', 'with', 'your', 'about', 'after', 'again', 'could', 'first', 'great', 'other', 'should', 'think', 'through', 'water', 'where', 'which', 'world', 'would', 'always', 'because', 'between', 'different', 'everything', 'something', 'sometimes', 'together', 'without', 'questions', 'should', 'reschedule', 'wants', 'poor', 'remain', 'rich', 'richer', 'michael', 'socialist', 'it'}:
            continue
            
        for ending in GEORGIAN_ENDINGS:
            if word.endswith(ending):
                ending_count += 1
                break
    
    if ending_count >= 6:  # Increased from 5
        return "ka_en"
    
    # Signal 3: Georgian stopwords (increased threshold)
    stopword_count = sum(1 for word in words if word in GEORGIAN_STOPWORDS)
    if stopword_count >= 2:  # Lowered from 4
        return "ka_en"
    
    # Signal 4: Check for Georgian phonetic patterns (more specific)
    # Only count patterns that are very characteristic of Georgian
    # These patterns should be extremely rare in English
    georgian_phonetic_patterns = [
        r'\b[a-z]*[aeiou][aeiou][aeiou][aeiou][aeiou][aeiou][a-z]*\b',  # Sextuple vowels (extremely Georgian)
        r'\b[a-z]*[aeiou][aeiou][aeiou][aeiou][aeiou][aeiou][aeiou][a-z]*\b',  # Septuple vowels (extremely Georgian)
    ]
    
    phonetic_score = 0
    for pattern in georgian_phonetic_patterns:
        matches = re.findall(pattern, text_lower)
        phonetic_score += len(matches)
    
    if phonetic_score >= len(words) * 0.1:  # Very low threshold for extremely rare patterns
        return "ka_en"
    
    # Signal 5: Check for Georgian-specific word patterns (more specific)
    # Only count very distinctive Georgian patterns
    georgian_word_patterns = [
        r'\b[a-z]*[aeiou][aeiou][aeiou][aeiou][aeiou][aeiou][a-z]*\b',  # Sextuple vowels
    ]
    
    pattern_score = 0
    for pattern in georgian_word_patterns:
        matches = re.findall(pattern, text_lower)
        pattern_score += len(matches)
    
    if pattern_score >= len(words) * 0.1:  # Very low threshold for extremely rare patterns
        return "ka_en"
    
    # If we get here, it's likely English but didn't meet strict criteria
    # Check if it's very short or has mixed signals
    if len(words) <= 5:
        # For very short texts, be more lenient with English
        if english_ratio >= 0.5:
            return "en"
        elif any(word in GEORGIAN_STOPWORDS for word in words):
            return "ka_en"
        else:
            return "unknown"
    
    # For longer texts, check if there are any Georgian signals before defaulting to English
    if any(word in GEORGIAN_STOPWORDS for word in words):
        return "ka_en"
    
    # Check for clearly Georgian words that might not be in stopwords
    clearly_georgian_words = {
        'gamarjoba', 'nakhvamdis', 'madloba', 'arafris', 'gamarjobat', 'saazgvargaret',
        'kopnisas', 'internetit', 'sargebloba', 'shesadzlebelia', 'operatorebis',
        'kselshi', 'romeltanac', 'jeoselsdadebuli', 'roumingis', 'xelshekruleba',
        'benshi', 'yoveltvis', 'tavs', 'verc', 'magaze', 'davdeb', 'magram',
        'mgoni', 'ikneba', 'goodwillshi', 'getyvi', 'elenes', 'naswavli',
        'ekonomikit', 'wigni', 'gadamishlia', 'advili', 'dzan', 'arafri',
        'pirveli', 'testia', 'madloba', 'didi', 'aseti', 'reakciistvis',
        'ravi', 'shemdegshi', 'rame', 'uketesi', 'gavakete', 'sheileba'
        # Removed 'chakapuli' as it appears in English context
    }
    
    # Check if any words match (clean punctuation first)
    matching_georgian_words = []
    for word in words:
        # Clean punctuation from word
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in clearly_georgian_words:
            matching_georgian_words.append(word)
    
    # Debug: print what we found
    if matching_georgian_words:
        print(f"Found clearly Georgian words: {matching_georgian_words}")
        return "ka_en"
    
    print(f"No clearly Georgian words found. All words: {words}")
    
    # Default to English for longer texts that didn't meet strict criteria
    # This is a fallback for texts that are probably English but didn't hit all markers
    return "en"

def looks_forward_or_headers(t: str) -> bool:
    return bool(FORWARD_HDR.search(t) or HEADER_BLOCK.search(t))

def looks_receipt_or_system(t: str) -> bool:
    """
    Relaxed: require stronger evidence before dropping as receipt/system.
    Keep conversational content that merely mentions numbers or a single cue.
    """
    # Strong receipt/system cues must be paired with length or numeric density
    if RECEIPT_CUES.search(t):
        n = len(t)
        if n >= 300:
            return True
        digits = len(MANY_NUMS.findall(t))
        if n >= 180 and (digits / max(n, 1)) > 0.35:
            return True
        # Otherwise, treat as conversational mention
        return False
    # numeric density alone is not enough unless extremely high and long
    n = len(t)
    if n >= 300:
        digits = len(MANY_NUMS.findall(t))
        if (digits / max(n, 1)) > 0.45:
            return True
    # excessive divider lines: raise thresholds to avoid false positives
    if t.count('----') >= 3 or t.count('::') >= 4:
        return True
    return False

def is_noise_text(t: str) -> bool:
    t = t.strip()
    # Keep very short acknowledgements for context; only drop truly empty/one-char
    if len(t) < 2: 
        return True
    if looks_forward_or_headers(t): 
        return True
    if looks_receipt_or_system(t): 
        return True
    return False

def is_mostly_media(text):
    # Strip timestamps/names first
    t = re.sub(r'\[\d{1,2}/\d{1,2}/\d{2,4},.*?\]', '', text)
    t = re.sub(r'[A-Z][a-z]+ [A-Z][a-z]+:', '', text)  # naive name removal
    # Tokenize and see if >80% are placeholders
    tokens = t.split()
    media_tokens = sum(1 for tok in tokens if MEDIA_PLACEHOLDER.search(tok))
    # relax threshold to keep more chat content even with a media reference
    return tokens and (media_tokens / len(tokens) > 0.95)

def clean_whatsapp_artifacts(t: str) -> str:
    """Remove WhatsApp-specific tails like
    "[2/13/23, 8:46:17 PM] Name Surname: GIF omitted" and similar placeholders.
    Also removes orphan media placeholders if present.
    """
    if not t:
        return t
    # Remove timestamp + name + any '... omitted' segments (not only at tail)
    t = re.sub(
        r"\s*\[\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\]\s*[^:]{1,100}:\s*[^\n]*?omitted\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # Remove orphan media/contact/document/location placeholders if any remain
    t = re.sub(
        r"\b(?:image|video|audio|sticker|gif|document|contact(?:\s+card)?|location|photo|picture|file)\s+omitted\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def strip_quotes_and_sigs(t):
    """Comprehensive email cleaning function."""
    if not t:
        return t
    
    # Remove email quote patterns
    for pattern in EMAIL_QUOTE_PATTERNS:
        t = re.sub(pattern, '', t, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove tech artifacts
    for pattern in TECH_ARTIFACTS:
        t = re.sub(pattern, '', t, flags=re.IGNORECASE)
    
    # Remove HTML entities
    t = re.sub(r'&nbsp;', ' ', t)  # Non-breaking spaces
    t = re.sub(r'&amp;', '&', t)   # Ampersands
    t = re.sub(r'&lt;', '<', t)    # Less than
    t = re.sub(r'&gt;', '>', t)    # Greater than
    t = re.sub(r'&quot;', '"', t)  # Quotes
    
    # Remove quote markers and headers
    t = re.sub(r'(?m)^\s*>\s*', '', t)  # Remove quote markers at line start
    
    # General pattern to catch "On ... wrote:" headers in various formats
    t = re.sub(r'On .+? wrote:', '', t, flags=re.IGNORECASE)
    
    # Fix missing spaces after punctuation
    t = re.sub(r'([,\.!?])([A-Za-z])', r'\1 \2', t)  # Add space after comma, period, exclamation, question mark if followed by letter
    
    # Remove repetitive signature blocks (common in forwarded emails)
    # Look for repeated patterns like "Name - ID University Year College"
    signature_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+ - [a-z]+\d+ [A-Z][a-z]+ University \d{4} College of [A-Z][a-z]+ [A-Z][a-z]+)'
    t = re.sub(signature_pattern, '', t)
    
    # Remove signatures at the end of messages (including Georgian ones)
    # Remove "-- Name" patterns at the end
    t = re.sub(r'\s*--\s*[ა-ჰ]+\s*[ა-ჰ]*\s*$', '', t)  # Georgian signatures
    t = re.sub(r'\s*--\s*[A-Za-z]+\s*[A-Za-z]*\s*$', '', t)  # English signatures
    
    # Remove standalone Georgian names like "[Name]" at the end
    t = re.sub(r'\s*[ა-ჰ]+\.[ა-ჰ]+\s*$', '', t)
    
    # Remove common email closings with names at the end
    t = re.sub(r'\s*Best,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Best, [Name]"
    t = re.sub(r'\s*Thanks,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Thanks, [Name]"
    t = re.sub(r'\s*Thank you,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Thank you, [Name]"
    t = re.sub(r'\s*Best regards,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Best regards, [Name]"
    t = re.sub(r'\s*Regards,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Regards, [Name]"
    t = re.sub(r'\s*Sincerely,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Sincerely, [Name]"
    t = re.sub(r'\s*Cheers,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Cheers, [Name]"
    t = re.sub(r'\s*Yours,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Yours, [Name]"
    t = re.sub(r'\s*Kind regards,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Kind regards, [Name]"
    t = re.sub(r'\s*Warm regards,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Warm regards, [Name]"
    t = re.sub(r'\s*Take care,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Take care, [Name]"
    t = re.sub(r'\s*All the best,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "All the best, [Name]"
    t = re.sub(r'\s*Best wishes,\s*[A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Best wishes, [Name]"
    
    # Remove more complex closings with names (not just at the end)
    t = re.sub(r'\s*Thank you for consideration,\s*[A-Za-z]+\s*[A-Za-z]*\s*$', '', t, flags=re.IGNORECASE)  # "Thank you for consideration, [Name]"
    t = re.sub(r'\s*Thank you for your time,\s*[A-Za-z]+\s*[A-Za-z]*\s*$', '', t, flags=re.IGNORECASE)  # "Thank you for your time, [Name]"
    t = re.sub(r'\s*Thank you for your help,\s*[A-Za-z]+\s*[A-Za-z]*\s*$', '', t, flags=re.IGNORECASE)  # "Thank you for your help, [Name]"
    t = re.sub(r'\s*Thank you for everything,\s*[A-Za-z]+\s*[A-Za-z]*\s*$', '', t, flags=re.IGNORECASE)  # "Thank you for everything, [Name]"
    t = re.sub(r'\s*Thank you for your time and attention,\s*[A-Za-z]+\s*[A-Za-z]*\s*$', '', t, flags=re.IGNORECASE)  # "Thank you for your time and attention, [Name]"
    
    # Remove closings with additional identifiers (like student IDs)
    t = re.sub(r'\s*[A-Za-z]+ [A-Za-z]+ - \d+ - [a-z]+\d+\s*$', '', t)  # "Name - ID - username"
    t = re.sub(r'\s*[A-Za-z]+ [A-Za-z]+ - [a-z]+\d+\s*$', '', t)  # "Name - username"
    t = re.sub(r'\s*[A-Za-z]+ [A-Za-z]+ - \d+\s*$', '', t)  # "Name - ID"
    
    # Remove closings with any text followed by full name and identifiers
    t = re.sub(r'\s*[^\s]+ [A-Za-z]+ [A-Za-z]+ - [a-z]+\d+ [A-Za-z]+ University \d{4} College of [A-Za-z]+ [A-Za-z]+\s*$', '', t)  # "Text Name - ID University Year College"
    t = re.sub(r'\s*[^\s]+ [A-Za-z]+ [A-Za-z]+ [A-Za-z]+ University \d{4} College of [A-Za-z]+ [A-Za-z]+\s*$', '', t)  # "Text Name University Year College"
    
    # Remove complex closings with full names and identifiers
    t = re.sub(r'\s*Thank you for your time and attention,\s*[A-Za-z]+ [A-Za-z]+ - [a-z]+\d+ [A-Za-z]+ University \d{4} College of [A-Za-z]+ [A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Thank you for your time and attention, Name - ID University Year College"
    t = re.sub(r'\s*Thank you for consideration,\s*[A-Za-z]+ [A-Za-z]+ [A-Za-z]+ University \d{4} College of [A-Za-z]+ [A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Thank you for consideration, Name University Year College"
    
    # Remove closings with "Sincerely" and full names
    t = re.sub(r'\s*Sincerely\s*[A-Za-z]+ [A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Sincerely [Name]"
    t = re.sub(r'\s*Sincerely,\s*[A-Za-z]+ [A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Sincerely, [Name]"
    
    # Remove closings with "Thank you again for your time" and full names
    t = re.sub(r'\s*Thank you again for your time,\s*[A-Za-z]+ [A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Thank you again for your time, [Name]"
    t = re.sub(r'\s*Thank you again for your time,\s*[A-Za-z]+ [A-Za-z]+ - [a-z]+\d+ [A-Za-z]+ University \d{4} College of [A-Za-z]+ [A-Za-z]+\s*$', '', t, flags=re.IGNORECASE)  # "Thank you again for your time, Name - ID University Year College"
    
    # Remove Georgian closings with names
    t = re.sub(r'\s*პატივისცემით,\s*[ა-ჰ]+\s*[ა-ჰ]*\s*$', '', t)  # "პატივისცემით, [Name]"
    t = re.sub(r'\s*ძალიან მადლობა,\s*[ა-ჰ]+\s*[ა-ჰ]*\s*$', '', t)  # "ძალიან მადლობა, [Name]"
    t = re.sub(r'\s*მადლობა,\s*[ა-ჰ]+\s*[ა-ჰ]*\s*$', '', t)  # "მადლობა, [Name]"
    
    # Remove repetitive signature blocks (common in forwarded emails)
    # Look for patterns like "Name - ID University Year College"
    t = re.sub(r'\s*[A-Z][a-z]+ [A-Z][a-z]+ - [a-z]+\d+ [A-Z][a-z]+ University \d{4} College of [A-Z][a-z]+ [A-Z][a-z]+\s*$', '', t)
    
    # Remove university signature blocks (common in academic emails)
    t = re.sub(r'\s*[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+ University \d{4} College of [A-Z][a-z]+ [A-Z][a-z]+\s*$', '', t)  # "Name University Year College"
    
    # Clean up extra whitespace
    t = re.sub(r'\s+', ' ', t)
    t = t.strip()
    
    return t

def clean_common_artifacts(t):
    """Remove common artifacts that appear across all platforms."""
    if not t:
        return t
    
    # Remove "<This message was edited>" strings (common in messaging apps)
    t = re.sub(r'\s*<This message was edited>\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*<This message was edited\.>\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*<Message was edited>\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*<Message was edited\.>\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*<Edited>\s*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*<Edited\.>\s*', ' ', t, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    t = re.sub(r'\s+', ' ', t)
    t = t.strip()
    
    return t

def clean_gmail_artifacts(t):
    """Additional Gmail-specific cleaning for signatures and artifacts."""
    if not t:
        return t
    
    # Remove Gmail mobile signatures
    t = re.sub(r'\s*Sent from my (iPhone|iPad|Android|mobile device)\s*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*Get Outlook for (iOS|Android)\s*$', '', t, flags=re.IGNORECASE)
    
    # Remove Gmail-specific patterns like "> > >" chains
    t = re.sub(r'\s*>\s*>\s*>\s*$', '', t)
    t = re.sub(r'\s*>\s*>\s*$', '', t)
    
    # Remove university email signatures with IDs (like "an476 Cornell University 2017 College of Arts and Sciences Economics")
    t = re.sub(r'\s*[A-Za-z]+\d+\s+[A-Za-z]+\s+University\s+\d{4}\s+College\s+of\s+[A-Za-z\s]+\s*$', '', t, flags=re.IGNORECASE)
    
    # Remove patterns like "Name - ID University Year College"
    t = re.sub(r'\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s*-\s*[a-z]+\d+\s+[A-Z][a-z]+\s+University\s+\d{4}\s+College\s+of\s+[A-Za-z\s]+\s*$', '', t)
    
    # Remove Gmail forwarded message artifacts
    t = re.sub(r'\s*----------\s*Forwarded message\s*----------\s*$', '', t, flags=re.IGNORECASE)
    
    # Remove Gmail quote markers that might have slipped through
    t = re.sub(r'\s*>\s*[A-Za-z\s]+\s*>\s*$', '', t)
    
    # Remove remaining university signature patterns
    t = re.sub(r'\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s+University\s+\d{4}\s*$', '', t)
    
    # Remove GMT timestamps and timezone info
    t = re.sub(r'\s*GMT[+-]\d{2}:\d{2}\s*[A-Za-z\s]*\s*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*\d{1,2}:\d{2}\s*GMT[+-]\d{2}:\d{2}\s*[A-Za-z\s]*\s*$', '', t, flags=re.IGNORECASE)
    
    # Remove virus-free messages and antivirus signatures
    t = re.sub(r'\s*Virus-free\.\s*www\.\s*avast\.com\s*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*Virus-free\s*$', '', t, flags=re.IGNORECASE)
    
    # Remove UUIDs and hash-like strings
    t = re.sub(r'\s*<#[A-F0-9-]{36}>\s*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}\s*$', '', t, flags=re.IGNORECASE)
    
    # Remove email addresses in angle brackets
    t = re.sub(r'\s*<[^>]+@[^>]+>\s*$', '', t)
    
    # Remove quote chains with text
    t = re.sub(r'\s*>\s*[^>]+\s*>\s*$', '', t)
    t = re.sub(r'\s*>\s*[^>]+\s*$', '', t)
    
    # Remove date patterns like "2011/6/9"
    t = re.sub(r'\s*\d{4}/\d{1,2}/\d{1,2}\s*$', '', t)
    
    # Remove company names at the end
    t = re.sub(r'\s*[A-Za-z]+\s+Sales\s*$', '', t)
    
    # Remove standalone dashes and separators
    t = re.sub(r'\s*--\s*$', '', t)
    t = re.sub(r'\s*<>\s*$', '', t)
    
    # Remove partial timestamps and timezone info
    t = re.sub(r'\s*:\d{1,2}\s*GMT[+-]\d{2}:\d{2}\s*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*GMT[+-]\d{2}:\d{2}\s*$', '', t, flags=re.IGNORECASE)
    
    # Remove incomplete email addresses and partial signatures
    t = re.sub(r'\s*<\s*$', '', t)
    t = re.sub(r'\s*[A-Za-z]+\s+[A-Za-z]+\s*<\s*$', '', t)
    
    # Remove remaining signature patterns with names and dashes
    t = re.sub(r'\s*[A-Za-z]+\s+[A-Za-z]+\s*-\s*$', '', t)
    t = re.sub(r'\s*[A-Za-z]+\s+[A-Za-z]+\s*[A-Za-z]*\s*$', '', t)
    
    # Remove date patterns at the end
    t = re.sub(r'\s*\d{4}/\d{1,2}/\d{1,2}\s*[A-Za-z\s]*\s*$', '', t)
    
    # Remove any remaining artifacts with colons and timestamps
    t = re.sub(r'\s*:\d{1,2}\s*$', '', t)
    
    # Clean up any remaining artifacts
    t = re.sub(r'\s+', ' ', t)
    t = t.strip()
    
    return t

def ensure_min_schema(r):
    """Light normalization; assume parsers already did heavy lifting"""
    r.setdefault("platform", "unknown")
    r.setdefault("conversation_id", "unknown")
    r.setdefault("message_id", "")
    r.setdefault("date", "")
    r.setdefault("turn_role", "partner")
    r.setdefault("direction", "inbound")
    r.setdefault("body_text", r.get("body_raw") or "")
    r.setdefault("lang", "unknown")  # Will be overridden
    r.setdefault("is_reply", r.get("is_reply", False))
    
    # Strip email quotes/signatures and platform-specific artifacts
    if r["body_text"]:
        # First clean common artifacts that appear across all platforms
        r["body_text"] = clean_common_artifacts(r["body_text"])
        r["body_text"] = strip_quotes_and_sigs(r["body_text"])
        if (r.get("platform") or "").lower() == "whatsapp":
            r["body_text"] = clean_whatsapp_artifacts(r["body_text"])
        elif (r.get("platform") or "").lower() == "gmail":
            r["body_text"] = clean_gmail_artifacts(r["body_text"])
    
    # Reclassify language using our logic
    r["lang"] = classify_text(r["body_text"])
    
    r["date"] = iso(r["date"])
    r["conversation_key"] = convo_key(r)
    r["conversation_hash"] = conv_hash(r["conversation_key"])
    return r

def iso(dt):
    """Accept already-ISO strings; returns comparable ISO"""
    if not dt: 
        return ""
    return unicodedata.normalize("NFC", dt)

def convo_key(rec):
    return f"{rec.get('platform','')}::{rec.get('conversation_id','')}"

def conv_hash(key):
    return hashlib.blake2b(key.encode('utf-8'), digest_size=8).hexdigest()

def near_dup_key(text):
    norm = re.sub(r'\s+', ' ', re.sub(r'\W+', ' ', (text or "").lower())).strip()
    return hashlib.blake2b(norm.encode('utf-8'), digest_size=12).hexdigest()

def load_jsonl_many(paths):
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                yield json.loads(line)

def build_pairs(records, k_ctx=6, min_target_chars=3, trivial_replies=None):
    """Build conversation pairs with proper language classification and cleanup"""
    trivial_replies = trivial_replies or {"ok","k","kk","thanks","thank you","ty","cool","nice","got it","noted","sure"}
    by_thread = defaultdict(list)
    for r in records:
        by_thread[r["conversation_hash"]].append(r)

    for th in by_thread.values():
        th.sort(key=lambda x: x["date"])
        ctx = deque(maxlen=k_ctx)
        last_partner_nonempty = None  # allow minimal context fallback
        for r in th:
            ctx.append({"role": r["turn_role"], "text": r["body_text"]})
            if r["turn_role"] == "partner":
                bt = (r.get("body_text") or "").strip()
                if bt:
                    last_partner_nonempty = {"role": "partner", "text": bt}
            if r["turn_role"] == "you":
                target = r["body_text"].strip()
                if is_noise_text(target): 
                    continue
                if is_mostly_media(target):
                    continue
                if len(target) < min_target_chars: 
                    continue
                if target.lower() in trivial_replies:
                    continue
                
                context = [c for c in list(ctx)[:-1] if not is_noise_text(c["text"])]
                # If all prior turns were considered noise, allow a minimal fallback
                if not context and last_partner_nonempty is not None:
                    context = [last_partner_nonempty]
                if not context:
                    continue
                    
                yield {
                    "id": f"pair-{r['message_id'] or r['conversation_hash']}-{r.get('thread_index','')}",
                    "context": context,
                    "target_text": target,
                    "meta": {
                        "platform": r["platform"],
                        "conversation_hash": r["conversation_hash"],
                        "date": r["date"],
                        "lang": r["lang"],
                        "len_chars": len(target)
                    }
                }

def build_mono(records, min_chars=80):
    """Build mono examples with proper language classification and cleanup"""
    for r in records:
        if r["turn_role"] != "you":
            continue
        txt = (r["body_text"] or "").strip()
        if is_noise_text(txt):
            continue
        if is_mostly_media(txt):
            continue
        if len(txt) < min_chars:
            # Retain informative short texts: at least 5 words or 30+ chars
            if not (len(txt) >= 30 or len(txt.split()) >= 5):
                continue
        if not txt:
            continue
        yield {
            "id": f"mono-{r['message_id'] or r['conversation_hash']}-{r.get('thread_index','')}",
            "text": txt,
            "meta": {
                "platform": r["platform"],
                "conversation_hash": r["conversation_hash"],
                "date": r["date"],
                "lang": r["lang"],
                "len_chars": len(txt),
                "subject_norm": r.get("subject_norm")
            }
        }

def split_by_conversation(convo_ids, train=0.8, val=0.1, seed=42):
    """Split conversations into train/val/test"""
    ids = list(convo_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train)
    n_val = int(n * val)
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]

def summarize(records, pairs, monos):
    """Generate statistics"""
    s = {
        "records_total": len(records),
        "by_platform": defaultdict(int),
        "by_lang": defaultdict(int),
        "unique_conversations": len({r["conversation_hash"] for r in records}),
        "pairs_total": len(pairs),
        "mono_total": len(monos),
        "avg_pair_target_len": 0.0,
        "avg_mono_len": 0.0,
    }
    for r in records:
        s["by_platform"][r["platform"]] += 1
        s["by_lang"][r["lang"]] += 1
    if pairs:
        s["avg_pair_target_len"] = sum(p["meta"]["len_chars"] for p in pairs) / len(pairs)
    if monos:
        s["avg_mono_len"] = sum(m["meta"]["len_chars"] for m in monos) / len(monos)
    s["by_platform"] = dict(s["by_platform"])
    s["by_lang"] = dict(s["by_lang"])
    return s

def main(args):
    in_paths = [str(p) for pat in args.inputs for p in Path().glob(pat)] if args.glob else args.inputs
    if not in_paths:
        raise SystemExit("No input files found.")
    
    # Load + normalize + reclassify + global dedup
    seen_dups = set()
    records = []
    for rec in load_jsonl_many(in_paths):
        rec = ensure_min_schema(rec)
        if not rec["body_text"]:
            continue
        
        k = near_dup_key(rec["body_text"] + "|" + rec["turn_role"])
        if k in seen_dups:
            continue
        seen_dups.add(k)
        records.append(rec)

    # Build datasets
    pairs = list(build_pairs(records, k_ctx=args.k, min_target_chars=args.min_target))
    monos = list(build_mono(records, min_chars=args.min_mono))

    # Conversation-level splits
    convo_ids = {r["conversation_hash"] for r in records}
    tr_ids, va_ids, te_ids = split_by_conversation(convo_ids, train=args.train_frac, val=args.val_frac, seed=args.seed)

    def part_of(split_ids, item):
        h = item["meta"]["conversation_hash"] if "meta" in item else item["conversation_hash"]
        return h in split_ids

    # Create separate directories for each language bucket
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create language-specific directories
    ka_dir = outdir / "georgian_unicode"
    ka_en_dir = outdir / "georgian_english_keyboard"
    en_dir = outdir / "english"
    
    for dir_path in [ka_dir, ka_en_dir, en_dir]:
        dir_path.mkdir(exist_ok=True)

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # Split by language and write to separate directories
    for lang_code, lang_dir in [("ka", ka_dir), ("ka_en", ka_en_dir), ("en", en_dir)]:
        # Filter pairs by language
        lang_pairs = [p for p in pairs if p["meta"]["lang"] == lang_code]
        lang_monos = [m for m in monos if m["meta"]["lang"] == lang_code]
        
        # Split into train/val/test
        lang_convo_ids = {p["meta"]["conversation_hash"] for p in lang_pairs}
        if lang_convo_ids:
            tr_lang, va_lang, te_lang = split_by_conversation(lang_convo_ids, train=args.train_frac, val=args.val_frac, seed=args.seed)
        else:
            tr_lang, va_lang, te_lang = [], [], []
        
        # Write language-specific files
        write_jsonl(lang_dir / "pairs_train.jsonl", [p for p in lang_pairs if part_of(tr_lang, p)])
        write_jsonl(lang_dir / "pairs_val.jsonl", [p for p in lang_pairs if part_of(va_lang, p)])
        write_jsonl(lang_dir / "pairs_test.jsonl", [p for p in lang_pairs if part_of(te_lang, p)])
        
        write_jsonl(lang_dir / "mono_train.jsonl", [m for m in lang_monos if part_of(tr_lang, m)])
        write_jsonl(lang_dir / "mono_val.jsonl", [m for m in lang_monos if part_of(va_lang, m)])
        write_jsonl(lang_dir / "mono_test.jsonl", [m for m in lang_monos if part_of(te_lang, m)])
        
        # Write stats for each language
        lang_stats = {
            "language": lang_code,
            "pairs_total": len(lang_pairs),
            "mono_total": len(lang_monos),
            "train_val_test_counts": {
                "pairs": {
                    "train": len([p for p in lang_pairs if part_of(tr_lang, p)]),
                    "val": len([p for p in lang_pairs if part_of(va_lang, p)]),
                    "test": len([p for p in lang_pairs if part_of(te_lang, p)]),
                },
                "mono": {
                    "train": len([m for m in lang_monos if part_of(tr_lang, m)]),
                    "val": len([m for m in lang_monos if part_of(va_lang, m)]),
                    "test": len([m for m in lang_monos if part_of(te_lang, m)]),
                }
            }
        }
        with open(lang_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(lang_stats, f, ensure_ascii=False, indent=2)

    # Overall stats
    stats = summarize(records, pairs, monos)
    stats["inputs"] = in_paths
    stats["k_context"] = args.k
    stats["min_target_chars"] = args.min_target
    stats["min_mono_chars"] = args.min_mono
    stats["train_val_test_counts"] = {
        "pairs": {
            "train": sum(1 for p in pairs if part_of(tr_ids, p)),
            "val": sum(1 for p in pairs if part_of(va_ids, p)),
            "test": sum(1 for p in pairs if part_of(te_ids, p)),
        },
        "mono": {
            "train": sum(1 for m in monos if part_of(tr_ids, m)),
            "val": sum(1 for m in monos if part_of(va_ids, m)),
            "test": sum(1 for m in monos if part_of(te_ids, m)),
        }
    }
    
    with open(outdir / "overall_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote to {outdir.resolve()}")
    print(f"Language buckets created:")
    print(f"  - Georgian (Unicode): {ka_dir}")
    print(f"  - Georgian (English Keyboard): {ka_en_dir}")
    print(f"  - English: {en_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess with proper three-bucket language classification.")
    ap.add_argument("--inputs", nargs="+", required=True, help="JSONL files (or globs if --glob).")
    ap.add_argument("--glob", action="store_true", help="Treat --inputs as glob patterns.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("-k", type=int, default=6, help="Max context turns for pairs.")
    ap.add_argument("--min-target", type=int, default=3, help="Min target char length for pairs.")
    ap.add_argument("--min-mono", type=int, default=40, help="Min char length for mono (relaxed from 80).")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction by conversation.")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Val fraction by conversation.")
    ap.add_argument("--seed", type=int, default=42, help="Split seed.")
    args = ap.parse_args()
    main(args)
