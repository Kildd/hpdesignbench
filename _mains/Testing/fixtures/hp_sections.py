from matplotlib import pyplot as plt

# from core.visualization_core.visualization import plot_cross_section
from _mains.Testing.fixtures.materials import concrete_c50_uls, solidian_Q142_pre_50, \
    concrete_c80_uls, solidian_Q142, solidian_Q95_pre_50, solidian_Q95_pre_20, solidian_Q142_pre_60, concrete_c55_uls, \
    ref_solidian_Q85_pre_37, concrete_c30_uls, solidian_Q142_pre_75
from core.visualization_core.visualization import plot_cross_section
from slab_construction.slabs.hp_slab.hp_model.hp_geometry import HPGeometry
from slab_construction.slabs.hp_slab.hp_model.hp_shell import HPShell

"""
Author: Elliot Melcer

HP-Shells for Testing. Naming convention after:

    Jamila Loutfi (2023):   "Entwicklung eines parametrischen Bemessungsmodells 
                            für ein Deckensystem aus vorgespannten doppelt-gekrümmten 
                            Schalen aus Carbonbeton"
                            
"""

# --- HP Shells for Verification of Cracking and Ultimate Moments---

x = 0.5 # Location of Section in Girder

# HP Shell Geometries

# Note: Hx was determined through trial and error, since it was not provided in Loutfi (2023)
hp_c1_1         = HPGeometry(B = 1200, L = 6750, Hx = 40,  Hy = 160, t = 50, dy = 100, nt = 7)
hp_c1_2         = HPGeometry(B = 1200, L = 6750, Hx = 75,  Hy = 300, t = 70, dy = 50, nt = 8)
hp_c1_3         = HPGeometry(B = 1500, L = 6750, Hx = 125, Hy = 500, t = 50, dy = 50, nt = 1)
hp_c1_4         = HPGeometry(B = 1200, L = 6750, Hx = 100, Hy = 400, t = 100, dy = 80, nt = 10)
hp_ref          = HPGeometry(B = 1200, L = 8000, Hx = 80,  Hy = 320, t = 45, dy = 220, nt = 11)
hp_kappa_0_m2   = HPGeometry(B = 1520, L = 7000, Hx = 130, Hy = 520, t = 85, dy = 190, nt = 15)

hp_shell_c1_1_uls     = HPShell(hp_c1_1,            concrete_c50_uls, solidian_Q95_pre_20 ,  reinf_area =  50,   name = "C.1. Section 1")
hp_shell_c1_2_c50_uls = HPShell(hp_c1_2,            concrete_c50_uls, solidian_Q95_pre_50 ,  reinf_area =  50,   name = "C.1. Section 2.1")
hp_shell_c1_2_c80_uls = HPShell(hp_c1_2,            concrete_c80_uls, solidian_Q95_pre_50 ,  reinf_area =  50,   name = "C.1. Section 2.2")
hp_shell_c1_3_uls     = HPShell(hp_c1_3,            concrete_c50_uls, solidian_Q142       ,  reinf_area = 300,   name = "C.1. Section 3")
hp_shell_c1_4_uls     = HPShell(hp_c1_4,            concrete_c55_uls, solidian_Q142_pre_50,  reinf_area =  80,   name = "C.1. Section 4")
hp_shell_ref          = HPShell(hp_ref, concrete_c50_uls, ref_solidian_Q85_pre_37, reinf_area =   8.5, name ="Reference HP-Shell")
hp_shell_kappa_0_m2 = HPShell(hp_kappa_0_m2 ,   concrete_c30_uls, solidian_Q142_pre_75 , reinf_area = 100.5,   name = "HP-Shell uses Fallback Method for kappa_0")

# ULS Sections
hp_section_c1_1_uls     = hp_shell_c1_1_uls.section_at(x)
hp_section_c1_2_c50_uls = hp_shell_c1_2_c50_uls.section_at(x)
hp_section_c1_2_c80_uls = hp_shell_c1_2_c80_uls.section_at(x)
hp_section_c1_3_uls     = hp_shell_c1_3_uls.section_at(x)
hp_section_c1_4_uls     = hp_shell_c1_4_uls.section_at(x)
hp_section_kappa_0_m2 = hp_shell_kappa_0_m2.section_at(x)

# --- HP Shell for Verification of Moment-Curvature-Diagram ---
hp_c2 = HPGeometry(B = 1200, L = 6750, Hx = 70, Hy = 280, t = 40, dy = 30, nt = 5)

hp_shell_c2_uls = HPShell(hp_c2, concrete_c50_uls, solidian_Q142_pre_60, reinf_area = 10)

# ULS Sections
hp_section_c2_uls_x_0_00 = hp_shell_c2_uls.section_at(0.00, name = "C.2. Section at 0.00 * L")
hp_section_c2_uls_x_0_10 = hp_shell_c2_uls.section_at(0.10, name = "C.2. Section at 0.10 * L")
hp_section_c2_uls_x_0_20 = hp_shell_c2_uls.section_at(0.20, name = "C.2. Section at 0.20 * L")
hp_section_c2_uls_x_0_30 = hp_shell_c2_uls.section_at(0.30, name = "C.2. Section at 0.30 * L")
hp_section_c2_uls_x_0_40 = hp_shell_c2_uls.section_at(0.40, name = "C.2. Section at 0.40 * L")
hp_section_c2_uls_x_0_50 = hp_shell_c2_uls.section_at(0.50, name = "C.2. Section at 0.50 * L")

# Reference Section
hp_section_ref_x_0_50 = hp_shell_ref.section_at(0.50, name = "Reference Section at 0.50 * L")


if __name__ == "__main__":
    plot_cross_section(hp_section_c1_1_uls)
    plot_cross_section(hp_section_c1_2_c50_uls)
    plot_cross_section(hp_section_c1_2_c80_uls)
    plot_cross_section(hp_section_c1_3_uls)
    plot_cross_section(hp_section_c1_4_uls)

    plot_cross_section(hp_section_c2_uls_x_0_00)
    plot_cross_section(hp_section_c2_uls_x_0_10)
    plot_cross_section(hp_section_c2_uls_x_0_20)
    plot_cross_section(hp_section_c2_uls_x_0_30)
    plot_cross_section(hp_section_c2_uls_x_0_40)
    plot_cross_section(hp_section_c2_uls_x_0_50)

    plt.show()



