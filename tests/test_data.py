import gc
import pytest

def test_base_class_not_instantiable():
    """GELOSDataSet is abstract and cannot be instantiated directly."""
    from gelos.gelosdataset import GELOSDataSet

    with pytest.raises(TypeError):
        GELOSDataSet(
            bands={"S2L2A": ["RED"]},
            all_band_names={"S2L2A": ["RED", "GREEN", "BLUE"]},
        )
    
    gc.collect()

