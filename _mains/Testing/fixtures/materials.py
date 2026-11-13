"""
Author: Elliot Melcer
This File serves to provide Materials for Testing Purposes
"""

from structuralcodes import set_design_code
from structuralcodes.core.base import Material
from structuralcodes.materials.concrete import create_concrete
from structuralcodes.materials.constitutive_laws import Elastic
from structuralcodes.materials.reinforcement import create_reinforcement

from slab_construction.floor import InsulationMaterial, ScreedMaterial, InfillMaterial

set_design_code('ec2_2004')

# --- Material Definitions ---

# -- Concretes --


# - Definitions -
concrete_c30_uls = create_concrete(fck=30, constitutive_law ='parabolarectangle', alpha_cc = 0.85, gamma_c = 1.5, name ="C30/37 ULS")


concrete_c40_uls = create_concrete(fck=40, constitutive_law ='parabolarectangle', alpha_cc = 0.85, gamma_c = 1.5, name ="C40/50 ULS")
# concrete_c40_sls = create_concrete(fck=40, constitutive_law ='sargin', name ="C40/50 SLS")

concrete_c50_uls = create_concrete(fck=50, constitutive_law ='parabolarectangle', alpha_cc = 0.85, gamma_c = 1.5, name ="C50/60 ULS")
# concrete_c50_sls = create_concrete(fck=50, constitutive_law ='sargin', name ="C50/60 SLS")

concrete_c55_uls = create_concrete(fck=55, constitutive_law ='parabolarectangle', alpha_cc = 0.85, gamma_c = 1.5, name ="C55/67 ULS")
# concrete_c55_sls = create_concrete(fck=55, constitutive_law ='sargin', name ="C55/67 SLS")

concrete_c80_uls = create_concrete(fck=80, constitutive_law ='parabolarectangle', alpha_cc = 0.85, gamma_c = 1.5, name ="C80/95 ULS")
# concrete_c80_sls = create_concrete(fck=80, constitutive_law ='sargin', name ="C80/95 SLS")



# -- Steel Reinforcement --

# - Properties -
fyk_b500 = 500
ftk_b500 = 550
Es_b500 = 200000
epsuk_b500 = 0.07

# - Definition
reinforcement_B500 = create_reinforcement(
    fyk=fyk_b500,
    Es=Es_b500,
    ftk=ftk_b500,
    epsuk=epsuk_b500,
    density=7850,
    constitutive_law="elasticperfectlyplastic")

# -- CFRP Reinforcement --

# - Properties

# solidian GRID Q142/142-CCE-25 regular
fyk_Q142 = 2200
ftk_Q142 = 2200
Es_Q142 = 220_000
epsuk_Q142= 10/1000 # = 10 promille
density_Q142 = 1800

# # solidian GRID Q85/85-CCE-25
fyk_Q85 = 2800
ftk_Q85 = 2800
Es_Q85 = 230_000
epsuk_Q85= 12.173913/1000 # = 12.17 promille
density_Q85 = 1300

# solidian GRID Q95/95-CCE-38
fyk_Q95 = 2800
ftk_Q95 = 2800
Es_Q95 = 230_000
epsuk_Q95= 12.173913/1000 # = 12.17 promille
density_Q95 = 1340

# - Constitutive Law -

# solidian GRID Q142/142-CCE-25 regular
brittle_elastic_law_Q142 = Elastic(Es_Q142, eps_u=epsuk_Q142, name = "ElasticLaw")

# # solidian GRID Q85/85-CCE-25
brittle_elastic_law_Q85 = Elastic(Es_Q85, eps_u=epsuk_Q85,  name = "ElasticLaw")

# solidian GRID Q95/95-CCE-38
brittle_elastic_law_Q95 = Elastic(Es_Q95, eps_u=epsuk_Q95,  name = "ElasticLaw")


# - Material Definitions -

# solidian GRID Q85/85-CCE-21 prestressed 37 %

ref_solidian_Q85_pre_37 = create_reinforcement(
    fyk=fyk_Q85,
    Es=Es_Q85,
    ftk=ftk_Q85,
    epsuk=epsuk_Q85,
    density=1340,
    constitutive_law=brittle_elastic_law_Q85,
    initial_strain= 0.37 * epsuk_Q85,
    gamma_s=1.3,
    gamma_eps = 1/1.3,
    name="solidian GRID Q85/85-CCE-21 prestressed 37 %"
)


# solidian GRID Q95/95-CCE-38 prestressed 20 %

solidian_Q95_pre_20 = create_reinforcement(
    fyk=fyk_Q95,
    Es=Es_Q95,
    ftk=ftk_Q95,
    epsuk=epsuk_Q95,
    density=density_Q95,
    constitutive_law=brittle_elastic_law_Q95,
    initial_strain= 0.20 * epsuk_Q95,
    gamma_s=1.3,
    gamma_eps = 1/1.3,
    name="solidian GRID Q95/95-CCE-38 prestressed 20%"
)

# solidian GRID Q95/95-CCE-38 prestressed 50 %

solidian_Q95_pre_50 = create_reinforcement(
    fyk=fyk_Q95,
    Es=Es_Q95,
    ftk=ftk_Q95,
    epsuk=epsuk_Q95,
    density=density_Q95,
    constitutive_law=brittle_elastic_law_Q95,
    initial_strain= 0.50 * epsuk_Q95,
    gamma_s=1.3,
    gamma_eps = 1/1.3,
    name="solidian GRID Q95/95-CCE-38 prestressed 50%"
)

# solidian GRID Q142/142-CCE-25 regular

solidian_Q142 = create_reinforcement(
    fyk=fyk_Q142,
    Es=Es_Q142,
    ftk=ftk_Q142,
    epsuk=epsuk_Q142,
    density=density_Q142,
    constitutive_law=brittle_elastic_law_Q142,
    gamma_s = 1.3,
    gamma_eps = 1/1.3,
    name = "solidian GRID Q142/142-CCE-25"
)

# solidian GRID Q142/142-CCE-25 prestressed 50 %

solidian_Q142_pre_50 = create_reinforcement(
    fyk=fyk_Q142,
    Es=Es_Q142,
    ftk=ftk_Q142,
    epsuk=epsuk_Q142,
    density=density_Q142,
    constitutive_law = brittle_elastic_law_Q142,
    initial_strain = 0.50 * epsuk_Q142,
    gamma_s = 1.3,
    gamma_eps = 1/1.3,
    name = "solidian GRID Q142/142-CCE-25 prestressed 50%"
)

# solidian GRID Q142/142-CCE-25 prestressed 75 %

solidian_Q142_pre_75 = create_reinforcement(
    fyk=fyk_Q142,
    Es=Es_Q142,
    ftk=ftk_Q142,
    epsuk=epsuk_Q142,
    density=density_Q142,
    constitutive_law = brittle_elastic_law_Q142,
    initial_strain = 0.75 * epsuk_Q142,
    gamma_s = 1.3,
    gamma_eps = 1/1.3,
    name = "solidian GRID Q142/142-CCE-25 prestressed 50%"
)

# solidian GRID Q142/142-CCE-25 prestressed 60 %

solidian_Q142_pre_60 = create_reinforcement(
    fyk=fyk_Q142,
    Es=Es_Q142,
    ftk=ftk_Q142,
    epsuk=epsuk_Q142,
    density=density_Q142,
    constitutive_law = brittle_elastic_law_Q142,
    initial_strain = 0.60 * epsuk_Q142,
    gamma_s = 1.3,
    gamma_eps = 1/1.3,
    name = "solidian GRID Q142/142-CCE-25 prestressed 60%"
)

# -- Floor Materials --
# Like in Ausbau.xlsx
infill = InfillMaterial(density = 1000, name = "Infill")
sound_insulation = InsulationMaterial(density = 24, E_dyn = 0.6, name = "Sound insulation")
screed = ScreedMaterial(density = 2000, name = "Screed")

# Line in Tabelle 7-2 in Loutfi (2023)
ref_infill = InfillMaterial(density = 1000, name = "Infill")
ref_sound_insulation = InsulationMaterial(density = 150, E_dyn = 30., name = "Sound insulation")
ref_screed = ScreedMaterial(density = 2100, name = "Screed")

