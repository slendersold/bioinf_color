import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
import colour
from colour import (MSDS_CMFS, SDS_ILLUMINANTS, MultiSpectralDistributions,
                    SpectralShape, gamma_function, msds_to_XYZ)
from colour.models.rgb import RGB_COLOURSPACES, XYZ_to_RGB

default_colref_basepath = '.'
def load_msds(config_path, wl_min, wl_max, wl_step):
    wvl_df = pd.read_csv(config_path)
    wvl_df['Wavelength'] = (wvl_df['Wavelength']
                            .astype(np.uint16)
                            )
    assert wvl_df.Wavelength.iloc[
               0] == wl_min, "passed min value of wavelength does not match csv stored at passed base path"
    assert wvl_df.Wavelength.iloc[
               -1] == wl_max, "passed max value of wavelength does not match csv stored at passed base path"
    assert wvl_df.Wavelength.diff().mode().item() == wl_step, "passed step value of wavelengths does not match csv stored at passed base path"

    chart_sd = wvl_df.set_index('Wavelength').drop(['CA'], axis=1)

    return (chart_sd.columns.to_list(),
            MultiSpectralDistributions(chart_sd.to_numpy(),
                                       SpectralShape(wl_min, wl_max, wl_step))
            )


def calculate_xyz_chart(msds,
                        observer=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
                        illuminant=SDS_ILLUMINANTS['D65']):
    xyzs = msds_to_XYZ(msds,
                       cmfs=observer,
                       illuminant=illuminant,
                       method='Integration'
                       ) / 100
    assert xyzs.max() < 1
    return xyzs


def calculate_rgb_chart(xyz_chart, colorspace='sRGB'):
    assert colorspace in ['sRGB', 'NTSC (1987)', 'DON RGB 4']

    _colspace = RGB_COLOURSPACES[colorspace]
    if colorspace == 'NTSC (1987)':  # 'NTSC (1987)', 1953
        _colspace._cctf_decoding = partial(gamma_function, exponent=1.8)
        _colspace._cctf_encoding = partial(gamma_function, exponent=1 / 1.8)

    ret = np.zeros_like(xyz_chart)
    for idx, p in enumerate(xyz_chart):
        ret[idx] = XYZ_to_RGB(
            XYZ=p,
            colourspace=_colspace,
            apply_cctf_encoding=True
        )
    ret = np.nan_to_num(ret)
    ret[ret < 0] = 0
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference_basepath',
        type=str,
        default=f"{default_colref_basepath}/calibration_data",
    )
    parser.add_argument(
        '--wl_range',
        nargs='+',
        type=int,
        default=(340, 830, 5),
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args = parse_args()
    # reference_basepath = args.reference_basepath
    # wl_min, wl_max, wl_step = args.wl_range
    reference_basepath = './calibration_data/'
    wl_min, wl_max, wl_step = 340, 830, 5

    ref_patch_order, msds = load_msds(f'{reference_basepath}/wavelengths_{wl_step}nmstep.csv',
                                      wl_min, wl_max, wl_step)

    reference_d50 = calculate_xyz_chart(msds,
                                        observer=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
                                        illuminant=SDS_ILLUMINANTS['D50'])
    reference_d65 = calculate_xyz_chart(msds,
                                        observer=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
                                        illuminant=SDS_ILLUMINANTS['D65'])

    reference_srgbs = calculate_rgb_chart(reference_d65, 'sRGB')
    reference_don4s = calculate_rgb_chart(reference_d50, 'DON RGB 4')
    reference_ntscs = calculate_rgb_chart(reference_d65, 'NTSC (1987)')

    np.savez(f"{reference_basepath}/xyz_values.npz",
             D50=reference_d50,
             D65=reference_d65)
    print(f"generated D65 and D50 XYZ values for 2-degree observer, saved to {reference_basepath}/xyz_values.npz")

    np.savez(f"{reference_basepath}/rgb_values.npz",
             sRGB=reference_srgbs,
             DON4=reference_don4s,
             NTSC=reference_ntscs)
    print(f"generated sRGB, DON RGB 4 and NTSC values, saved to {reference_basepath}/rgb_values.npz")
    print(reference_srgbs)