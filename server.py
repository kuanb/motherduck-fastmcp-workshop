"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import re
from collections import Counter, defaultdict
from statistics import median
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import caller_email

mcp = FastMCP("sf-311")
logger = get_logger("sf-311")

HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}

CountGroup = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
MedianGroup = Literal["neighborhood", "agency"]


def _mask_address(address: str) -> str:
    """Turn "994 HARRISON ST, SAN FRANCISCO, ..." into "900 block of HARRISON ST".

    Addresses without a house number, such as intersections, stay as they are.
    """
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """Return a request's fields with the address masked to the block."""
    row = request.as_dict()
    row["address"] = _mask_address(request.address)
    return row


def _check_name(value: str | None, choices: list[str], kind: str) -> None:
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean {', '.join(repr(s) for s in suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} {value!r}.{hint} Call dataset_info for the exact {kind} names.")


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_name(service, SERVICES, "service")
    _check_name(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: Annotated[int, Field(ge=1, le=MAX_ROWS)] = 20,
) -> list[dict]:
    """List individual 311 requests, newest first, to see examples or find request IDs.

    Each row has the request's id, opened and closed times, status, service,
    subtype, agency, neighborhood, local_name, address (only to the block,
    for example "900 block of HARRISON ST"), and source.
    Returns at most 50 rows, so do not use this to count or to compute
    statistics: use count_requests for "how many" and median_close_hours for
    "how long". Use get_request when you already have a request_id.
    service and neighborhood must be exact names from dataset_info.
    """
    rows = sorted(_matching(service, neighborhood, status), key=lambda r: r.opened, reverse=True)
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: CountGroup | None = None,
) -> int | dict[str, int]:
    """Count 311 requests this week, optionally filtered and broken down.

    Without group_by, returns one number. With group_by, returns a dict of
    group to count, largest first, at most 50 groups; requests with a blank
    value for that field are left out. weekday groups by the day the request
    was opened. Use this for any "how many" or "which has the most" question
    instead of searching and counting rows yourself. For time to close, use
    median_close_hours. service and neighborhood must be exact names from
    dataset_info.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def requests_by_hour(
    service: str | None = None,
    neighborhood: str | None = None,
) -> dict:
    """Count 311 requests by the hour of day they were opened, in San Francisco local time.

    Returns {"busiest_hour": h, "counts": {"0": n, ..., "23": n}} with all 24
    hours, including zeros; busiest_hour is the hour with the most requests
    (the earliest hour wins a tie) or null if nothing matches. Use this for
    "what time of day" or "which hour" questions. For totals or breakdowns
    by neighborhood, service, source, or weekday, use count_requests.
    service and neighborhood must be exact names from dataset_info.
    """
    counts = Counter(r.opened.hour for r in _matching(service, neighborhood))
    busiest = max(range(24), key=lambda h: (counts[h], -h)) if counts else None
    return {"busiest_hour": busiest, "counts": {str(h): counts[h] for h in range(24)}}


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: str | None = None,
    group_by: MedianGroup | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed 311 requests, rounded to 0.1.

    Only closed requests count; open ones are ignored. Without group_by,
    returns one number (or null if nothing matching has closed). With
    group_by="neighborhood" or "agency", returns a dict of group to median
    hours, fastest first, leaving out groups with fewer than 10 closed
    requests. Use this for "how long does it take" questions; use
    count_requests for volumes. service must be an exact name from
    dataset_info.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {g: round(median(v), 1) for g, v in hours.items() if len(v) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return every field of one 311 request by its request_id.

    The address is given only to the block, for example "900 block of HARRISON ST".

    Use this when you already have an id, for example from search_requests.
    To find requests by service, neighborhood, or status, use search_requests.
    """
    logger.info("get_request request_id=%s caller=%s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find valid request IDs.")
    return _public(request)


if __name__ == "__main__":
    mcp.run()
