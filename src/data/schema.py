"""CMeEE-V2 字段与下标约定。"""

ENTITY_TYPES = ("dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep")
REQUIRED_ENTITY_FIELDS = ("start_idx", "end_idx", "type", "entity")


def entity_key(entity: dict) -> tuple:
    """严格匹配使用的三元组：左闭右开下标加类型。"""
    return (entity["start_idx"], entity["end_idx"], entity["type"])
