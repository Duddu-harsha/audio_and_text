# Configuration file for Video Content Analyzer
# All hardcoded values moved here for better maintainability

# OCR Configuration
OCR_LANGUAGES = ['en']
OCR_CONFIDENCE_THRESHOLD = 0.4
TESSERACT_CONFIDENCE_THRESHOLD = 40
TESSERACT_CONFIG = r'--oem 3 --psm 6'

# Video Processing Configuration
MAX_VIDEO_WIDTH = 1280
FRAME_EXTRACTION_FPS = {
    'short_video': 6,    # < 5 seconds
    'medium_video': 3,   # 5-15 seconds
    'long_video': 1      # > 15 seconds
}
VIDEO_DURATION_THRESHOLDS = {
    'short': 5,
    'medium': 15
}

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = 'pcm_s16le'

# Sentiment Analysis Configuration
SENTIMENT_MODELS = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "distilbert-base-uncased-finetuned-sst-2-english"
]
SENTIMENT_TEXT_MAX_LENGTH = 512
SENTIMENT_SCORE_THRESHOLDS = {
    'positive': 0.1,
    'negative': -0.1
}

# Whisper Configuration
DEFAULT_WHISPER_MODEL = "base"
WHISPER_CONFIDENCE_DEFAULT = 0.8

# Text Processing Configuration
MIN_TEXT_LENGTH = 1
SIMILARITY_THRESHOLD = 0.8
TEXT_MERGE_MIN_LENGTH = 2

# FFmpeg Configuration
FFMPEG_AUDIO_EXTRACT_CMD = [
    'ffmpeg', '-i', '{input}', 
    '-vn', '-acodec', 'pcm_s16le', 
    '-ar', '16000', '-ac', '1',
    '-y', '{output}'
]
FFMPEG_DURATION_CMD = [
    'ffprobe', '-v', 'quiet', '-show_entries', 
    'format=duration', '-of', 'csv=p=0', '{input}'
]

# Enhanced Toxicity Word Lists
TOXIC_WORDS = {
    'english': {
        'high_severity': [
            'fuck', 'fucking', 'shit', 'nigger', 'faggot', 'cunt', 'bitch', 
            'motherfucker', 'cocksucker', 'bastard', 'asshole', 'wanker', 
            'twat', 'bugger', 'bollocks', 'clusterfuck', 'chicken shit', 
            'coonass', 'cornhole', 'cox', 'goddamnit', 'jack-ass', 'anus', 
            'arse', 'arsehole', 'ass-hat', 'ass-jabber', 'ass-pirate', 
            'assbag', 'blowjob'
        ],
        'medium_severity': [
            'damn', 'hell', 'ass', 'piss', 'bastard', 'idiot', 'moron', 
            'stupid', 'loser', 'whore', 'arsehole', 'balls', 'bint', 
            'bitch', 'bollocks', 'bullshit', 'feck', 'munter', 'pissed', 
            'son of a bitch', 'tits', 'crap', 'dumb ass', 'dumb-ass', 
            'father-fucker', 'god damn', 'bloody', 'blooming', 'prick', 
            'douchebag', 'twit', 'cretin', 'wuss', 'scumbag'
        ],
        'ethnic_slurs': [
            'nigger', 'coon', 'chink', 'gook', 'kike', 'faggot', 'dyke', 
            'paki', 'raghead', 'spic', 'wog', 'injun', 'honky', 'ape', 
            'jungle bunny', 'towelhead', 'fairy', 'poof', 'redneck', 
            'cracker', 'abeed', 'abid', 'abbo', 'abe', 'pikey', 'gippo', 
            'golliwog', 'spastic', 'negro'
        ],
        'sexual_explicit': [
            'porn', 'cock', 'dick', 'pussy', 'tits', 'blowjob', 'slut', 
            'whore', 'jizz', 'sex', 'boner', 'muff', 'wank', 'cunt', 
            'shag', 'knob', 'bussy', 'boofing', 'facial', 'facefuck', 
            'felching', 'fisting', 'lavda', 'tatte', 'smash', 'thirsty', 
            'soggy biscuit', 'serosorting', 'soaking', 'sexed', 'sexfarm', 
            'sexhound', 'sexhouse', 'sexing', 'sexkitten', 'sexpot', 
            'sexslave', 'sextoy', 'sextoys', 'sexual', 'sexually', 
            'sexwhore', 'sexymoma', 'sexy-slim', 'shaggin'
        ]
    },
    'hindi': {
        'high_severity': [
            'madarchod', 'behenchod', 'chut', 'lund', 'gaand', 'randi', 
            'bhosdi', 'gandiye', 'goo', 'gu', 'gote', 'gotey', 'hag', 
            'haggu', 'hagne', 'harami', 'kutta', 'gadha', 'fattu', 
            'charsi', 'chooche', 'choochi', 'chuchi', 'klpd', 'jhantu', 
            'ckp', 'chilla chod', 'bhadwa', 'bhosdike', 'chutiya', 'mc', 
            'bc', 'gandu', 'chordo', 'chuche', 'gel saffa', 'ghanta', 
            'gundmare', 'kameenay'
        ],
        'medium_severity': [
            'harami', 'bevakoof', 'kutta', 'gadha', 'fattu', 'ullu', 
            'haraamkhor', 'moot', 'tatti', 'hijda', 'porkistan', 'moover', 
            'kuttey ki aulad', 'chup kar', 'ghasti', 'gashti', 'gasti', 
            'ghassad', 'pig', 'bakchodi', 'lau lahsun'
        ],
        'sexual_explicit': [
            'chut', 'lund', 'gaand', 'randi', 'bhosdi', 'lavda', 'tatte', 
            'buur', 'charsi', 'chooche', 'choochi', 'chuchi', 'madarchod', 
            'behenchod', 'beti chod', 'muth mar', 'burr', 'choot', 
            'chootiya', 'lavde', 'lundfakeer', 'muth', 'lund ka pani', 
            'lovejuice', 'cum', 'dickjuice'
        ]
    },
    'telugu': {
        'high_severity': [
            'erugu', 'bosudi', 'lanja kodka', 'modda', 'pooku', 'dengu', 
            'dhani puku lo erugu', 'doola lanja', 'erri pooka', 
            'gudha kindha kovvu pattindha', 'hasta prayogam chai', 
            'jinniappa', 'kammani pooku', 'lanjamunda', 'guddha naku', 
            'pedda salladhi', 'veri puka', 'pichi sanasi', 'madda guddu', 
            'donga na kodaka', 'nee mama barri', 'nee amma ni', 'lanja', 
            'gudda', 'nee amma', 'chetta na kodaka'
        ],
        'medium_severity': [
            'vedhava', 'sachinoda', 'kojja', 'gaadida', 'adda gaadida', 
            'geviri gaadida', 'vedava', 'yedava', 'pichoda', 'addagadidha', 
            'pandi', 'dunnapoti', 'dunnapota'
        ],
        'sexual_explicit': [
            'modda', 'pooku', 'sallu', 'gudda', 'vattalu', 'dengu', 
            'lanja', 'modda cheeku lanja', 'pooku naaku', 'gudda naaku', 
            'nee ammanu denga', 'doola lanja', 'erri pooka', 
            'gudha kindha kovvu pattindha', 'hasta prayogam chai', 
            'jinniappa', 'kammani pooku', 'guddha naku', 'pedda salladhi', 
            'veri puka', 'madda guddu'
        ]
    }
}

# False positive contexts for toxicity detection
FALSE_POSITIVE_CONTEXTS = {
    'ass': [
        'class', 'glass', 'mass', 'pass', 'grass', 'brass', 'assess', 
        'assignment', 'assistant', 'associate', 'assassin', 'bass', 
        'embassy', 'passion', 'classic', 'assistance', 'massive', 
        'asset', 'passage', 'assumption'
    ],
    'hell': [
        'hello', 'shell', 'bell', 'well', 'tell', 'sell', 'cell', 
        'hellish', 'michelle', 'shellfish', 'hellicopter', 'hellenic', 
        'helluva', 'shelley', 'hellen', 'hellman', 'hellfire', 
        'hellcat', 'hellion'
    ],
    'damn': [
        'damage', 'adamant', 'amsterdam', 'condemn', 'damned', 'damp', 
        'damsel', 'madam', 'dame', 'damascus', 'damnation', 'dampen', 
        'adam', 'random', 'pandamonium'
    ],
    'shit': [
        'shirt', 'shift', 'shitake', 'shiite', 'shito', 'shita', 
        'ashita', 'escheat', 'shitzu', 'shitless', 'shithead', 
        'mishit', 'shitty', 'bullshitake'
    ],
    'sex': [
        'essex', 'sussex', 'sextant', 'asexual', 'sexist', 'sextet', 
        'sussex', 'middlesex', 'wessex', 'sexagenarian', 'sexagesimal', 
        'sextuplet', 'sexism', 'sexless'
    ],
    'cum': [
        'circumference', 'cucumber', 'document', 'cumulative', 
        'cumbersome', 'scum', 'succumb', 'cumulus', 'cummerbund', 
        'encumber', 'cumbia', 'incumbent', 'cuming', 'cumbrance'
    ],
    'dic': [
        'dictionary', 'medical', 'indicate', 'predicament', 'dickens', 
        'dickinson', 'dickie', 'medic', 'dichotomy', 'dickhead', 
        'indict', 'medicaid', 'judicial', 'periodic'
    ],
    'tit': [
        'constitution', 'title', 'petition', 'titrate', 'titillate', 
        'titmouse', 'titbit', 'titanium', 'titular', 'prostitute', 
        'institute', 'substitute', 'titania', 'titian', 'titivate'
    ],
    'cunt': [
        'scunthorpe', 'cunctator', 'cuntz', 'account', 'cuntington', 
        'viscount', 'cunter', 'cuntline', 'cuntzville'
    ],
    'fuck': [
        'fucking awesome', 'holy fuck', 'fuck yeah', 'luck', 'fuchsia', 
        'fuckwit', 'fuckery', 'fuckton', 'fuckup', 'fuckhead'
    ],
    'bitch': [
        'habitual', 'bitching', 'pitch', 'bitchy', 'ambitious', 
        'bitchin', 'bitches brew', 'subitch', 'bitchface'
    ],
    'nigger': [
        'snigger', 'niggle', 'niggardly', 'niggard', 'niggerish', 
        'niggly', 'renigger', 'denigger'
    ],
    'cock': [
        'cockpit', 'cocktail', 'cockatoo', 'peacock', 'cockroach', 
        'cockney', 'cockerel', 'hancock', 'shuttlecock', 'cockles', 
        'cocksure'
    ]
}

# Safety level configuration
SAFETY_LEVELS = ["safe", "review_needed", "unsafe"]

# File extensions for video detection
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# Cleanup patterns for text processing
TEXT_CLEANUP_PATTERN = r'[^\w\s]'

# Audio analysis configuration
NO_SPEECH_PROB_DEFAULT = 0.2
SEGMENT_TEXT_MIN_LENGTH = 1

# Output configuration
OUTPUT_FILE_PREFIX = "complete_analysis_"
OUTPUT_DATE_FORMAT = "%Y%m%d_%H%M%S"