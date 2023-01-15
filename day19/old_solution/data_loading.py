from blueprint import Blueprint, OreRobot, ClayRobot, ObsidianRobot, GeodeRobot

INPUT_FILE_PATH = "../input.txt"  # TODO: do it better


def load_blueprints(filepath: str) -> list[Blueprint]:
    blueprints = []
    with open(INPUT_FILE_PATH) as file:
        lines = file.readlines()

        for line in lines:
            match line.split():
                case [
                    "Blueprint",
                    n,
                    "Each",
                    "ore",
                    "robot",
                    "costs",
                    or_o,
                    "ore.",
                    "Each",
                    "clay",
                    "robot",
                    "costs",
                    cr_o,
                    "ore.",
                    "Each",
                    "obsidian",
                    "robot",
                    "costs",
                    obr_o,
                    "ore",
                    "and",
                    obr_c,
                    "clay.",
                    "Each",
                    "geode",
                    "robot",
                    "costs",
                    gr_o,
                    "ore",
                    "and",
                    gr_ob,
                    "obsidian.",
                ]:
                    blueprint = Blueprint(
                        ore_robot=OreRobot(ore=int(or_o)),
                        clay_robot=ClayRobot(ore=int(cr_o)),
                        obsidian_robot=ObsidianRobot(ore=int(obr_o), clay=int(obr_c)),
                        geode_robot=GeodeRobot(ore=int(gr_o), obsidian=int(gr_ob)),
                        n=int(n[:-1])
                    )
                    blueprints.append(blueprint)

                case _:
                    raise IOError("Invalid blueprint input")
    return blueprints