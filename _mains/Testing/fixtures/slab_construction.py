from _mains.Testing.fixtures.floor import test_floor, ref_floor
from _mains.Testing.fixtures.hp_slabs import hp_slab_c1_4_uls, hp_slab_c1_3_uls, hp_slab_c1_2_c80_uls, \
    hp_slab_c1_2_c50_uls, hp_slab_c1_1_uls, hp_slab_ref
from slab_construction.slab_construction import SlabConstruction

test_slab_construction_c1_1     = SlabConstruction(hp_slab_c1_1_uls, test_floor)
test_slab_construction_c1_2_c50 = SlabConstruction(hp_slab_c1_2_c50_uls, test_floor)
test_slab_construction_c1_2_c80 = SlabConstruction(hp_slab_c1_2_c80_uls, test_floor)
test_slab_construction_c1_3     = SlabConstruction(hp_slab_c1_3_uls, test_floor)
test_slab_construction_c1_4     = SlabConstruction(hp_slab_c1_4_uls, test_floor)
test_slab_construction_ref      = SlabConstruction(hp_slab_ref, ref_floor)