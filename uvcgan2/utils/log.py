import logging

def reduce_pil_verbosity(level):
    """A hack to stop PIL dumping large amounts of useless DEBUG info"""
    logger = logging.getLogger('PIL')
    logger.setLevel(max(logging.WARNING, level))

def setup_logging(level = logging.DEBUG):
    """Setup logging."""
    logger = logging.getLogger()

    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s]: %(levelname)s %(message)s'
    )

    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    reduce_pil_verbosity(logger.level)
