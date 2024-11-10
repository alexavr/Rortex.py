# On Kubrick: conda activate tmp

import numpy as np  # conda install conda-forge::numpy
import pandas as pd # conda install anaconda::pandas
import xarray as xr # conda install -c conda-forge xarray dask netCDF4 bottleneck
import datetime
from pathlib import Path
from config import *

def test() -> None:

    print(f"Test Rortex settings and data avaliability")

def rortex(src_file:str) -> None:

    ts = datetime.datetime.now()

    print(" * Wrap for computing Rortex")

    tensor = read_tensor(src_file)
    omega = compute_omega(tensor)
    ss_img, ss_vec = compute_swirling_strength(tensor)
    r3d = compute_rortex(ss_img, ss_vec, omega)

    ss_img, _ = compute_swirling_strength(tensor, flat=True)
    r2d = compute_rortex(ss_img, _, omega, flat=True)

    output(r2d,r3d)


    print(f" * Time elapsed: {datetime.datetime.now()-ts}")

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

    print(datetime.datetime.now()-ts)

    return ds_out

def compute_omega(tensor):
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

    print(datetime.datetime.now()-ts)

    return ds_out

def compute_swirling_strength(tensor, flat=False):
    '''
        Расчет силы (мнимая часть) и направления (вектор) Силы Завихренности
        ССЫЛКА!

        flat = False -> 3D постановка
        flat = True  -> 2D постановка


    '''



    ts = datetime.datetime.now()

    if flat:
        print(f"{' * * computing Swirling Strength (2D)':40}", end='')

        tensor_np = np.array([
                        [tensor.dudx, tensor.dudx],
                        [tensor.dudy, tensor.dudy],
                    ])

    else:
        print(f"{' * * computing Swirling Strength (3D)':40}", end='')

        tensor_np = np.array([
                        [tensor.dudx, tensor.dudx, tensor.dudx],
                        [tensor.dudy, tensor.dudy, tensor.dudy],
                        [tensor.dudz, tensor.dudz, tensor.dudz],
                    ])

    tensor_np = tensor_np.transpose(2, 3, 4, 5, 0, 1)

    eigVal, eigVec = np.linalg.eig(tensor_np)

    A = np.random.random((4,3,3))
    eigenValues, eigenVectors = np.linalg.eig(A)
    print("\n")
    print(A.shape)
    print(eigenValues.shape)
    print(eigenVectors.shape)

    print("+++++++++++")
    idx = eigenValues.argsort()[::-1]
    print(idx.shape)
    eigenValues = eigenValues[::idx]
    print(eigenValues.shape)
    eigenVectors = eigenVectors[::idx]
    print(eigenVectors.shape)

    # print("\n")
    # print(tensor_np.shape)
    # print(eigVal.shape)
    # print(eigVec.shape)
    # print(eigVal[2,2,50,50,:])

    # idx = eigVal.argsort()[::-1]
    # print(idx.shape)
    # print(eigVal[::-1].shape)
    # # eigVal = eigVal[idx]
    # # eigVec = eigVec[:,idx]
    # # print(eigVal[2,2,50,50,:])

    # # for ii in np.arange(eig_val.shape[0]):
    # #     for jj in np.arange(eig_val.shape[1]):
    # #         for kk in np.arange(eig_val.shape[2]):
    # #             if np.any(eig_val[ii,jj,kk,:].imag != 0):
    # #                 print(eig_val[ii,jj,kk,:])

    del tensor_np

    # # ЗАЧЕМ??????
    # # Потому что собственных значений 3
    # # Нам нужно то, которое принадлежит действительно значному собственному значению
    # # Чтобы определить вектор r
    # order = np.argsort(np.imag(eig_val))
    # ss_vec = np.array(eig_vec)[:,order][:,1]
    # ss_vec = eig_vec


    # # ЗАЧЕМ?
    eig_val_im = eigVal - np.real(eigVal) # Выделяем только мнимую часть
    # # ЗАЧЕМ?
    ss_img = np.abs(np.imag(eig_val_im)[:,:,:,:,1])

    # # Может повредить дальнейшему расчету R, поэтому пока не применяю
    # ss_img[ss_img <= 0.] = np.nan # критерий > 0

    print(datetime.datetime.now()-ts)
    exit()

    return ss_img, eig_vec









def compute_rortex(ss_img, ss_vec, omega, flat=False):
    ts = datetime.datetime.now()

    if flat:
        print(f"{' * * computing Rortex (2D)':40}", end='')
    else:
        print(f"{' * * computing Rortex (3D)':40}", end='')

    ds_out = omega.omega_x

    print(datetime.datetime.now()-ts)

    return ds_out

def output(r2d, r3d) -> None:
    ts = datetime.datetime.now()

    time0 = pd.to_datetime(r3d.XTIME)[0]
    file_dst = f"{OUTPUT_PATH}/rortex_{time0.strftime('%Y-%m-%d_%H:%M:%S')}"

    r2d.name = "R2D"; r2d.attrs['units'] = "XXX";
    r3d.name = "R3D"; r3d.attrs['units'] = "XXX";

    ds_out = xr.merge([ r2d, r3d ])

    del ds_out.attrs['units']
    ds_out.attrs['created'] = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    ds_out.to_netcdf(f"{file_dst}", mode='w')


    print(f" * * Output to {file_dst} {datetime.datetime.now()-ts}")

    return None

def main():

    test()


if __name__ == "__main__":
    main()
