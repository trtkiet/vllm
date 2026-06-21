# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Any

import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import routing as instrumentator_routing
from starlette.routing import Match, Mount
from starlette.types import Scope

from vllm.v1.metrics.prometheus import get_prometheus_registry


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def _get_route_path(route: Any, scope: Scope) -> str | None:
    route_path = getattr(route, "path", None)
    if route_path is not None:
        return route_path

    route_match = getattr(route, "_match", None)
    if route_match is None:
        return None

    match, _, child_route, route_context = route_match(scope)
    if match == Match.NONE or child_route is None:
        return None

    route_path = getattr(route_context, "path", None)
    if route_path is not None:
        return route_path

    if child_route is route:
        return None
    return _get_route_path(child_route, scope)


def _get_route_name(
    scope: Scope,
    routes: list[Any],
    route_name: str | None = None,
) -> str | None:
    """Get route names for FastAPI route containers.

    prometheus-fastapi-instrumentator expects every route to expose `.path`.
    FastAPI 0.137 added `_IncludedRouter` containers that match requests but
    only expose the selected child route through their private matcher.
    """
    for route in routes:
        match, child_scope = route.matches(scope)
        if match == Match.FULL:
            route_path = _get_route_path(route, scope)
            if route_path is None:
                return None

            route_name = route_path
            child_scope = {**scope, **child_scope}
            if isinstance(route, Mount) and route.routes:
                child_route_name = _get_route_name(
                    child_scope, route.routes, route_name
                )
                if child_route_name is None:
                    route_name = None
                else:
                    route_name += child_route_name
            return route_name
        if match == Match.PARTIAL and route_name is None:
            route_name = _get_route_path(route, scope)
    return None


def _patch_instrumentator_routing() -> None:
    if instrumentator_routing._get_route_name is _get_route_name:
        return
    instrumentator_routing._get_route_name = _get_route_name


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()
    _patch_instrumentator_routing()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
