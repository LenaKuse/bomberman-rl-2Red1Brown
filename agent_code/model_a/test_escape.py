"""
Standalone, isolated test for get_escape_direction()/get_danger_zone().
No main.py, no GUI needed -- builds fake game_state dicts by hand and
draws a picture of each scenario (field, danger zone, bombs, agent,
proposed escape direction).

Run from inside this folder:
    python test_escape.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from callbacks import get_escape_direction, get_danger_zone


def make_field(size=17):
    """Empty board with a wall border, like the real game's 'field'."""
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = -1
    field[:, 0] = field[:, -1] = -1
    return field


def visualize_scenario(game_state, escape_dir, distance_to_safety, title="Escape scenario", save_path=None):
    """
    Draws the field (walls/crates/free tiles), the current danger zone
    (red overlay), the bombs (black circles with their timer), the agent
    (blue dot) and an arrow for the proposed escape direction (green).
    """
    field = game_state['field']
    width, height = field.shape
    danger = get_danger_zone(game_state)
    agent_x, agent_y = game_state['self'][3]

    tile_colors = {-1: "#3a3a3a", 0: "#f2f2f2", 1: "#c99b62"}  # wall, free, crate

    fig, ax = plt.subplots(figsize=(6, 6))

    for x in range(width):
        for y in range(height):
            ax.add_patch(patches.Rectangle(
                (x, y), 1, 1,
                facecolor=tile_colors.get(int(field[x][y]), "white"),
                edgecolor="lightgray", linewidth=0.5))

    for (x, y) in danger:
        ax.add_patch(patches.Rectangle(
            (x, y), 1, 1, facecolor="red", alpha=0.35, edgecolor=None))

    for (bx, by), timer in game_state['bombs']:
        ax.add_patch(patches.Circle((bx + 0.5, by + 0.5), 0.35, facecolor="black"))
        ax.text(bx + 0.5, by + 0.5, str(timer), ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    ax.add_patch(patches.Circle((agent_x + 0.5, agent_y + 0.5), 0.3,
                                 facecolor="royalblue", edgecolor="black", zorder=5))

    arrow_map = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
    if escape_dir in arrow_map:
        dx, dy = arrow_map[escape_dir]
        ax.arrow(agent_x + 0.5, agent_y + 0.5, dx * 0.8, dy * 0.8,
                  head_width=0.2, head_length=0.2,
                  fc="limegreen", ec="limegreen", linewidth=2, zorder=6)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()  # y grows downward, matching how 'field' is laid out
    ax.set_aspect("equal")
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.set_title(f"{title}\nget_escape_direction() -> '{escape_dir}' \ndistance to safety: {distance_to_safety}")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


def run_scenario(name, field, bombs, self_pos, explosion_map=None):
    game_state = {
        'field': field,
        'bombs': bombs,
        'explosion_map': explosion_map if explosion_map is not None else np.zeros_like(field),
        'coins': [],
        'self': ('test', 0, True, self_pos),
        'others': [],
    }
    direction, distance_to_safety = get_escape_direction(game_state)
    print(f"[{name}] agent at {self_pos}, bombs={bombs} -> '{direction}'")
    visualize_scenario(game_state, direction, distance_to_safety, title=name, save_path=f"runawaytest/{name}.png")
    return direction


if __name__ == "__main__":
    # Scenario 1: agent standing inside a bomb's blast -> should flee sideways
    field1 = make_field()
    run_scenario("scenario1_in_danger", field1, [((5, 5), 3)], (5, 7))

    # Scenario 2: same bomb, but a wall blocks the blast before it reaches the agent
    field2 = make_field()
    field2[5, 6] = -1
    run_scenario("scenario2_wall_blocks_blast", field2, [((5, 5), 3)], (5, 7))

    # Scenario 3: agent far away from any bomb -> should be 'SAFE'
    field3 = make_field()
    run_scenario("scenario3_safe", field3, [((5, 5), 3)], (12, 12))

    # Scenario 4: agent sealed in on all 4 sides, bomb on its own tile -> 'WAIT' (trapped)
    field4 = make_field()
    ax_, ay_ = 5, 5
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        field4[ax_ + dx, ay_ + dy] = -1
    run_scenario("scenario4_trapped", field4, [((ax_, ay_), 3)], (ax_, ay_))

    #Scenario 5: agent not sealed, but no safe way to escape -> 'WAIT'
    field5 = make_field()
    field5[2, :] = -1
    field5[1, 8:] = -1
    run_scenario("scenario5_trapped", field5, [((1, 4), 3)], (1, 1))

    # Scenario 6: same layout as scenario 2, but a crate instead of a wall --
    # crates do NOT stop the blast (they just get destroyed by it and the
    # blast keeps going, see Bomb.get_blast_coords in items.py), so unlike
    # scenario 2 this should NOT come back 'SAFE'
    field6 = make_field()
    field6[5, 6] = 1  # crate, not a wall
    run_scenario("scenario6_crate_does_not_block_blast", field6, [((5, 5), 3)], (5, 7))

    # Scenario 7: agent standing directly on top of the bomb, open field --
    # every immediate neighbour is still on the bomb's own row/column, so
    # the first step of the returned path may itself still be "dangerous"
    # right now; safety is only reached one further step later
    field7 = make_field()
    run_scenario("scenario7_on_top_of_bomb_open_field", field7, [((8, 8), 3)], (8, 8))

    # Scenario 8: two bombs at once, agent caught between them on the same row --
    # checks that get_danger_zone correctly unions the blast radii of
    # multiple simultaneous bombs rather than only considering one
    field8 = make_field()
    run_scenario("scenario8_two_bombs", field8, [((5, 5), 3), ((9, 5), 3)], (7, 5))

    # Scenario 9: three bombs at once, agent caught between them on the same row --
    # checks that get_danger_zone correctly unions the blast radii of
    # multiple simultaneous bombs rather than only considering one
    field8 = make_field()
    run_scenario("scenario9_three_bombs", field8, [((5, 5), 3), ((9, 5), 3), ((7, 6), 1)], (7, 5))
