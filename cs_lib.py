# Based on
# (1) Xu, W.; Gao, Y.; Deng, Y.; Liu, J.; Liu, C. An Explicit Expression for the Calculation of the Rortex Vector. Physics of Fluids 2019, 31 (9), 095102. https://doi.org/10.1063/1.5116374.
# and
# https://github.com/utaresearch/Liutex_OA/blob/main/Liutex_UTA/Liutex_Python/Liutex_UTA.py#L58

# On Kubrick: conda activate R

# Install rortex enviroment:
# conda env create --file=environment.yml

# Perfomance for 1 file:
# NAAD LoRes - 20 s
# NAAD HiRes - 10 min
# TC 10km    - 1hr


import numpy as np  # conda install conda-forge::numpy
import pandas as pd # conda install anaconda::pandas
import xarray as xr # conda install -c conda-forge xarray dask netCDF4 bottleneck
import xarray_einstats.linalg as xlinalg # conda install -c conda-forge xarray-einstats
import datetime
from pathlib import Path
# from config import *
import os

# import matplotlib.pyplot as plt

def test() -> None:

    print(f"Test for Rortex settings and data avaliability")

def rortex(OUTPUT_PATH:str, src_file:str) -> None:

    ts = datetime.datetime.now()

    print(f"File: {src_file}")

    tensor = read_tensor(src_file)

    # omega = compute_omega(tensor)
    
    # Rortax 3D
    r3dm, r3dv = compute_rortex(tensor)

    # Rortex 2D
    tensor["dwdx"] = tensor["dwdx"] * 0
    tensor["dwdy"] = tensor["dwdy"] * 0
    tensor["dwdz"] = tensor["dwdz"] * 0
    tensor["dvdz"] = tensor["dvdz"] * 0
    tensor["dudz"] = tensor["dudz"] * 0
    r2dm, r2dv = compute_rortex(tensor)

    save(OUTPUT_PATH, tensor,r2dm,r3dm)


    print(f" * Total time elapsed: {datetime.datetime.now()-ts}")

    return None


def read_tensor(src_file:str):
    ts = datetime.datetime.now()
    print(f"{' * * reading tensor':40}", end='')
    # print(f" * * Reading {Path(src_file).name} from {Path(src_file).resolve().parent} :: ", end='')

    with xr.open_dataset(src_file) as ds:

        ds_out = xr.merge([ ds.dudx,ds.dudy,ds.dudz,ds.dvdx,ds.dvdy,ds.dvdz,ds.dwdx,ds.dwdy,ds.dwdz ])
        # почему-то merge меняет порядок измерений, мне это не нравится, поэтому возвращаю исходный
        dim_names = list(ds.dwdz.sizes)
        ds_out = ds_out.transpose(*dim_names)

    print(f"| {datetime.datetime.now()-ts}")

    return ds_out

def compute_omega(tensor):
    '''
        Calculates the vorticity using tensor matrix
    '''

    ts = datetime.datetime.now()
    print(f"{' * * computing Omega':40}", end='')

    omega_x = tensor.dwdy - tensor.dvdz
    omega_y = tensor.dudz - tensor.dwdx
    omega_z = tensor.dvdx - tensor.dudy

    omega_x.name = 'omega_x'
    omega_y.name = 'omega_y'
    omega_z.name = 'omega_z'

    ds_out = xr.merge([ omega_x, omega_y, omega_z ])
    # почему-то merge меняет порядок измерений, мне это не нравится, поэтому возвращаю исходный
    dim_names = list(omega_x.sizes)
    ds_out = ds_out.transpose(*dim_names)

    print(f"| {datetime.datetime.now()-ts}")

    return ds_out


def compute_rortex(tensor_xr):
    '''
        Расчет действительной и мнимой части собственных векторов тензора

    '''

    ts = datetime.datetime.now()


    if len(tensor_xr.count()) == 4:

        tensor = np.array([
                        [tensor_xr.dudx,   tensor_xr.dudy ],
                        [tensor_xr.dvdx,   tensor_xr.dvdy ],
                    ])
        tensor = tensor.transpose(2, 3, 4, 5, 0, 1)

        vorticity = np.array([ tensor_xr.dvdx - tensor_xr.dudy   ])
        vorticity = vorticity.transpose(1, 2, 3, 4, 0)

        tensorRank = 2

    elif len(tensor_xr.count()) == 9:

        tensor = np.array([
                        [tensor_xr.dudx, tensor_xr.dudy, tensor_xr.dudz],
                        [tensor_xr.dvdx, tensor_xr.dvdy, tensor_xr.dvdz],
                        [tensor_xr.dwdx, tensor_xr.dwdy, tensor_xr.dwdz],
                    ])
        tensor = tensor.transpose(2, 3, 4, 5, 0, 1)
        # tensor = xr.DataArray(tensor, dims=('time', 'z', 'y', 'x', 'xyz', 'uvw'),coords={"xyz": ['x', 'y', 'z'], "uvw": ['u', 'v', 'w'] })

        vorticity = np.array([
            tensor_xr.dwdy - tensor_xr.dvdz,
            tensor_xr.dudz - tensor_xr.dwdx,
            tensor_xr.dvdx - tensor_xr.dudy ])
        vorticity = vorticity.transpose(1, 2, 3, 4, 0)

        tensorRank = 3

    else:
        stop(f"Tensor has size of {len(tensor_xr.count())} but has to be 4 or 9")


    # Индексация тензора идет tensor xyz (по х), uvw (по y)

    # print(f"{' * * Tensor rank is ':40}" + f"| {tensorRank}")

    print(f"{'tensor':>40}", tensor.shape )
    print(f"{' vorticity':>40}", vorticity.shape)

    tdim, zdim, ydim, xdim = tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]

    # получаем собственные значения и собственный вектор (нормализованный?)
    ts_tmp = datetime.datetime.now()
    lambdas, evector = xlinalg.eig(tensor, dims=(4, 5))
    print(f"{'   * compute_rortex3d: eig ':40}" + f"| {datetime.datetime.now()-ts_tmp}")
    print(f"{'lambdas':>40}",lambdas.shape)
    print(f"{'evector':>40}",evector.shape)

    # Чистим память
    del tensor

    # Маска (комплексное - не комплексное) узлов собственных значений
    mask_ci = np.where( lambdas.imag != 0, True, False )
    print(f"{' mask_ci':>40}", mask_ci.shape)

    # Маска узлов, где есть 1 реальное и 2 комплексно сопряженных значения
    # Там где нет комплексных значений -- нет и вращения (rortex = 0)
    # Там где есть пара комплексных значение, они сопряжены и устроены так:  (X+0j; Y+Zj; Y-Zj)
    # Тогда находим индексы действительных элементов (X+0j)
    # (пустым значениям присваиваем -1)
    indxs_lambda_rr = np.where( mask_ci.sum(axis=-1) == 2, np.abs(lambdas.imag).argmin(axis=-1), -1 )

    # Согласно представлению (X+0j; Y+Zj; Y-Zj), 
    # индексы комплексного элемента с положительной мнимой частью будут indxs_lambda_rr + 1
    indxs_lambda_ci = np.where( indxs_lambda_rr != -1, (indxs_lambda_rr+1)%tensorRank, -1 )
    print(f"{'indxs_lambda_rr':>40}", indxs_lambda_rr.shape )
    print(f"{'indxs_lambda_ci':>40}", indxs_lambda_ci.shape )


    # Получаем действительные значения собственного вектора evector 
    # squeeze -- чтобы избежать результирующего массива вида (...,1)
    evector_rr = np.squeeze(
        np.take_along_axis(evector.real, indxs_lambda_rr[..., np.newaxis, np.newaxis], axis=-1), axis=-1)
    print(f"{'evector_rr':>40}", evector_rr.shape )

    # Чистим память
    del evector

    # Теперь получаем мнимую часть комплексного собственного значения тензора
    # squeeze -- чтобы избежать результирующего массива вида (...,1)
    lambda_ci = np.squeeze(
        np.take_along_axis(lambdas.imag, indxs_lambda_ci[..., np.newaxis], axis=-1), axis=-1)
    print(f"{'lambda_ci':>40}", lambda_ci.shape )

    # Отмечаем направление вращения
    # и получаем нормализованныз вектор рортекса
    tmp_dot = np.einsum('...i,...i->...', vorticity, evector_rr)
    tmp_dot = tmp_dot[...,np.newaxis]
    r = np.where(tmp_dot < 0, -evector_rr, evector_rr)
    print(f"{'r ':>40}", r.shape )

    # Проекция векторов нормализованного рортекса на завихренность
    w_dot_r = np.einsum('...i,...i->...', vorticity, r)
    print(f"{'w_dot_r ':>40}", w_dot_r.shape )

    # Получаем величину вектора Rortex
    # (As long as the velocity gradient tensor has complex eigenvalues, the
    # expression in the square root is always positive)
    rortex_mag =  w_dot_r - np.sqrt( w_dot_r**2 - 4.0*lambda_ci.real**2 )
    rortex_vec = rortex_mag[...,np.newaxis] * r
    print(f"{'rortex_mag ':>40}", rortex_mag.shape )
    print(f"{'rortex_vec ':>40}", rortex_vec.shape )

    rortex_mag = np.where(lambda_ci == 0, np.nan, rortex_mag)

    # Печатаем дебаг
    if True:
        for iy in range(41,45):
            for ix in range(41,45):
                string1 = "".join(f"{fprint(x*10**5)}; " for x in lambdas[3,3,iy,ix,:])
                string2 = "".join(f"{fprint(x)}; " for x in evector_rr[3,3,iy,ix,:])
                string = string1 + " || " + string2
                string = string + f" || {mask_ci[3,3,iy,ix]}"
                string = string + f" || {(indxs_lambda_rr[3,3,iy,ix]):2d}"
                string = string + f" || {(indxs_lambda_ci[3,3,iy,ix]):2d}"
                string = string + f" || {fprint(lambda_ci[3,3,iy,ix],".15f")}"
                string = string + f" || {fprint(rortex_mag[3,3,iy,ix], ".15f")}"
                print(string)


    return rortex_mag, rortex_vec



# def compute_rortex(ss_img, ss_vec, omega):
#     ts = datetime.datetime.now()

#     # if flat:
#     #     print(f"{' * * computing Rortex (2D)':40}", end='')
#     # else:
#     #     print(f"{' * * computing Rortex (3D)':40}", end='')

#     ds_out = omega.omega_x

#     print(f"| {datetime.datetime.now()-ts}")

#     return ds_out

def save(OUTPUT_PATH:str, src, r2d, r3d) -> None:
    ts = datetime.datetime.now()

    time = pd.to_datetime(src.XTIME)
    time0 = pd.to_datetime(src.XTIME)[0]

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    file_dst = f"{OUTPUT_PATH}/rortex_{time0.strftime('%Y-%m-%d_%H:%M:%S')}"

    R2D_out = xr.DataArray(r2d, dims=('time', 'bottom_top', 'south_north', 'west_east'),
                                coords={'time': time })

    R3D_out = xr.DataArray(r3d, dims=('time', 'bottom_top', 'south_north', 'west_east'),
                                coords={'time': time })

    # XLAT_out = xr.DataArray(src.XLAT[0])
    # XLONG_out = xr.DataArray(src.XLONG[0])

    R2D_out.name = "R2D"; R2D_out.attrs['units'] = "XXX";
    R3D_out.name = "R3D"; R3D_out.attrs['units'] = "XXX";
    # XLAT_out.name = "XLAT"
    # XLONG_out.name = "XLONG"

    ds_out = xr.merge([ R2D_out, R3D_out ])
    # ds_out = xr.merge([ R2D_out, R3D_out, XLAT_out, XLONG_out ])

    del ds_out.attrs['units']
    ds_out.attrs['created'] = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    ds_out.to_netcdf(f"{file_dst}", mode='w')


    print(f" * * Output to {file_dst} {datetime.datetime.now()-ts}")

    return None

def fprint(value, format="4.1f"):
    """

    Print Complex Value
    Переводит комплексное значение в строку форматированной записи для печати

    """

    if np.isnan(value):
        return "NaN"

    if np.iscomplexobj(value):
        string = f"{(value.real):+{format}}{(value.imag):+{format}}j"
    else:
        string = f"{(value):+{format}}"

    return string

def main():

    test()


if __name__ == "__main__":
    main()
