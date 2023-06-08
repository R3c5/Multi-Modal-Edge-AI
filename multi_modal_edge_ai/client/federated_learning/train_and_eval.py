from typing import Dict

from flwr.common import Scalar
from pymongo.collection import Collection

from multi_modal_edge_ai.commons.model import Model


def train(model: Model, database_collection: Collection, config) -> tuple[int, Dict[str, Scalar]]:
    pass


def evaluate(model: Model, database_collection: Collection, config) -> tuple[int, Dict[str, Scalar]]:
    pass
