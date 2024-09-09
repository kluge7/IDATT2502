from typing import Literal

from .figure import Figure
from .gridspec import SubplotSpec
from .transforms import Bbox

class LayoutGrid:
    def __init__(
        self,
        parent: LayoutGrid | tuple[int, int, int, int] | None = None,
        parent_pos: tuple[range, range] | tuple[int, int] = (0, 0),
        parent_inner: bool = False,
        name: str = "",
        ncols: int = 1,
        nrows: int = 1,
        h_pad: float | None = None,
        w_pad: float | None = None,
        width_ratios: None | list[float] = None,
        height_ratios: None | list[float] = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def reset_margins(self) -> None: ...
    def add_constraints(self) -> None: ...
    def hard_constraints(self) -> None: ...
    def add_child(
        self,
        child: LayoutGrid,
        i: int | range = 0,
        j: int | range = 0,
    ) -> None: ...
    def parent_constraints(self) -> None: ...
    def grid_constraints(self) -> None: ...
    def edit_margin(self, todo: Literal["left", "right", "bottom", "top"], size: float, cell: int) -> None: ...
    def edit_margin_min(
        self,
        todo: Literal["left", "right", "bottom", "top"],
        size: float,
        cell: int = 0,
    ) -> None: ...
    def edit_margins(self, todo: Literal["left", "right", "bottom", "top"], size: float) -> None: ...
    def edit_all_margins_min(self, todo: Literal["left", "right", "bottom", "top"], size: float): ...
    def edit_outer_margin_mins(self, margin: dict, ss: SubplotSpec) -> None: ...
    def get_margins(self, todo, col) -> Bbox: ...
    def get_outer_bbox(self, rows=0, cols=0) -> Bbox: ...
    def get_inner_bbox(self, rows=0, cols=0) -> Bbox: ...
    def get_bbox_for_cb(self, rows=0, cols=0) -> Bbox: ...
    def get_left_margin_bbox(self, rows=0, cols=0) -> Bbox: ...
    def get_bottom_margin_bbox(self, rows=0, cols=0) -> Bbox: ...
    def get_right_margin_bbox(self, rows=0, cols=0) -> Bbox: ...
    def get_top_margin_bbox(self, rows=0, cols=0) -> Bbox: ...
    def update_variables(self) -> None: ...

def seq_id() -> str: ...
def print_children(lb) -> None: ...
def plot_children(fig: Figure, lg=None, level: int = 0, printit: bool = False) -> None: ...
