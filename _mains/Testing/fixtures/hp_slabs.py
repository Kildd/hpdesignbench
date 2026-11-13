from _mains.Testing.fixtures.hp_sections import hp_shell_c1_4_uls, hp_shell_c1_3_uls, hp_shell_c1_2_c80_uls, \
    hp_shell_c1_2_c50_uls, hp_shell_c1_1_uls, hp_shell_ref
from _mains.Testing.fixtures.materials import infill, ref_infill
from slab_construction.slabs.hp_slab.hp_model.hp_slab import HPSlab

hp_slab_c1_1_uls        = HPSlab(hp_shell_c1_1_uls, infill)
hp_slab_c1_2_c50_uls    = HPSlab(hp_shell_c1_2_c50_uls, infill)
hp_slab_c1_2_c80_uls    = HPSlab(hp_shell_c1_2_c80_uls, infill)
hp_slab_c1_3_uls        = HPSlab(hp_shell_c1_3_uls, infill)
hp_slab_c1_4_uls        = HPSlab(hp_shell_c1_4_uls, infill)
hp_slab_ref             = HPSlab(hp_shell_ref, ref_infill)

if __name__ == "__main__":
    print("self load = ", hp_slab_c1_4_uls.self_load())