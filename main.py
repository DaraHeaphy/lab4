import yaml

from os import path as opath

from mypylibs.optical_flow import OFConfig, run_optical_flow

ROOT_IN = 'in'
ROOT_OUT = 'out'
EXT = 'mp4'


def load_config(case: int) -> dict:
    """Optional per-video overrides in in/XX.yaml (e.g. `optical_flow: {win_size: 31}`)."""
    cfg_path = opath.join(ROOT_IN, f'{case:02}.yaml')
    if not opath.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def main(case: int):
    video_in = opath.join(ROOT_IN, f'{case:02}.{EXT}')
    if not opath.exists(video_in):
        raise FileNotFoundError(video_in)
    params = load_config(case)

    # Step 2: optical flow -> per-frame position / velocity measurements
    of_cfg = OFConfig.from_dict(params.get('optical_flow'))
    meas, fps = run_optical_flow(
        video_in,
        opath.join(ROOT_OUT, f'{case:02}_of.{EXT}'),
        of_cfg,
        csv_out=opath.join(ROOT_OUT, f'{case:02}_of.csv'),
    )
    n_valid = sum(m.valid for m in meas)
    print(f'[{case:02}] optical flow: {n_valid}/{len(meas)} frames with a measurement')


if __name__ == '__main__':

    case = 0
    while(True):
        case += 1
        try:
            main(case)
        except FileNotFoundError:
            break
