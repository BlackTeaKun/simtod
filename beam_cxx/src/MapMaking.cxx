#include "MapMaking.hpp"
#include "healpix_base.h"
#include "pointing.h"
#include <vector>

template<class _data_type, class _ptg_type>
MapMaking<_data_type, _ptg_type>::MapMaking(int _nside, bool _pol):
hb()
{
    nside = _nside;
    pol = _pol;
    hb.SetNside(nside, RING);

    tempmap = new MapBase<_data_type>(_nside, _pol);
    hitmap  = new MapBase<int>(_nside, false);
}

template<class _data_type, class _ptg_type>
MapMaking<_data_type, _ptg_type>::~MapMaking()
{
    tempmap->free_raw_data();
    delete tempmap;
}

template<class _data_type, class _ptg_type>
int MapMaking<_data_type, _ptg_type>::add_Scan(const Scan_data<_data_type, _ptg_type> &scan)
{
    int nsample = scan.nsample;
    std::vector<int> idx(scan.nsample);
    for (int i = 0; i < nsample; ++i){
        pointing ptg(scan.theta[i], scan.phi[i]);
        idx[i] = hb.ang2pix(ptg);
    }
    
    // For Hitmap
    hitmap->map_data(0, idx).array() += 1;

    
    // Eigen tod wrapper
    using _tod_type = Eigen::RowVector<_data_type, Eigen::Dynamic>;
    
    Eigen::Map<_tod_type> tod1(scan.tod1, nsample);
    Eigen::Map<_tod_type> tod2(scan.tod2, nsample);
    _tod_type pair_sum  = (tod1 + tod2) / 2;
    _tod_type pair_diff = (tod1 - tod2) / 2;

    // For Temperature
    tempmap->map_data(0, idx).array() += pair_sum;
    return 0;
}

template<class _data_type, class _ptg_type>
MapBase<_data_type> *MapMaking<_data_type, _ptg_type>::get_map()
{
    resmap->map_data = tempmap->map_data;
    resmap->map_data.row(0).array() /= hitmap->map_data.array();
    return resmap;
}

template<class _data_type, class _ptg_type>
MapBase<int> *MapMaking<_data_type, _ptg_type>::get_hitmap()
{
    return hitmap;
}
