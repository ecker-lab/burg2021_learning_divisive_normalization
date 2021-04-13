import datajoint as dj

from divisivenormalization.insilico import *

schema = dj.schema("mburg_dnrebuttal_insilico", locals())
dj.config["schema_name"] = "mburg_dnrebuttal_insilico"
dj.config["display.limit"] = 20
dj.config["enable_python_native_blobs"] = True

# Insert entries describing the fitted models into the Fit() table
Fit().insert(get_fit_entries(path_args.dn_path, 'dn', 10))
Fit().insert(get_fit_entries(path_args.nonspecific_path, 'dn-nonspecific', 10))
Fit().insert(get_fit_entries(path_args.subunit_path, 'subunit', 10))
Fit().insert(get_fit_entries(path_args.cnn3_path, 'cnn3', 10))

print("Optimal Gabor")
OptimalGabor().populate(reserve_jobs=True)

print("Orthogonal Plaids Contrast")
OrthPlaidsContrast().populate(reserve_jobs=True)

print("OrthPlaidsContrastIndex")
OrthPlaidsContrastIndex().populate(reserve_jobs=True)

print("OrthPlaidsContrastInhibPercentPhaseAvg")
OrthPlaidsContrastInhibPercentPhaseAvg().populate(reserve_jobs=True)

print("Center Surround rsponses")
CenterSurroundResponses().populate(reserve_jobs=True)

print("Center Surround Stats")
CenterSurroundStats().populate(reserve_jobs=True)

print("Done")
