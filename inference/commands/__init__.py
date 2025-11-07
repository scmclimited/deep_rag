"""
CLI command modules.
"""
from inference.commands.ingest import ingest
from inference.commands.query import query
from inference.commands.infer import infer
from inference.commands.query_graph import query_graph
from inference.commands.infer_graph import infer_graph
from inference.commands.health import health
from inference.commands.graph import graph
from inference.commands.inspect import inspect
from inference.commands.test import test, test_app

__all__ = [
    'ingest',
    'query',
    'infer',
    'query_graph',
    'infer_graph',
    'health',
    'graph',
    'inspect',
    'test',
    'test_app',
]

