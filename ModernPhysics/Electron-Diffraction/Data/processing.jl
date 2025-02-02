using DataFrames, CSV, GLM

const DATAFILES = ("./raw-data-spherical.tsv", "./raw-data-linear.tsv")
const PLANCK_CONSTANT = 4.135e-15
const MASS_ELECTRON = 0.511e6
const SPEED_OF_LIGHT = 2.99e8

wavelength(U::Float64) = PLANCK_CONSTANT * SPEED_OF_LIGHT / sqrt(U^2 + 2.0 * U * MASS_ELECTRON)

wavelength_σ(U::Float64, U_σ::Float64 = 0.1e3) = U_σ * wavelength(U) * (U + MASS_ELECTRON) / (U^2 + 2.0 * U * MASS_ELECTRON)

process_data(df_in::DataFrame) = DataFrame(
    "U (kV)" => df_in[!, "U (kV)"],
    "r1" => df_in[!, "2r1 (mm)"] ./ 2.0,
    "r2" => df_in[!, "2r2 (mm)"] ./ 2.0,
    "λ" => wavelength.(df_in[!, "U (kV)"] * 1e3) * 1e12,
    "λσ" => wavelength_σ.(df_in[!, "U (kV)"] * 1e3) * 1e12
)

weighted_linear_regression(df::DataFrame, formula) = coef(lm(formula, df, wts = 1.0 ./ (df.λσ .^ 2)))[2]

format_dataframe_for_output(df::DataFrame) = DataFrame(
    "U (kV)" => df[!, "U (kV)"],
    "r1 (mm)" => round.(df[!, "r1"], digits = 2),
    "r2 (mm)" => round.(df[!, "r2"], digits = 2),
    "λ (pm)" => round.(df[!, "λ"], digits = 1),
    "λσ (pm)" => round.(df[!, "λσ"], digits = 1)
)

main() = begin
    df_sp, df_li = (CSV.read(DATAFILES[1], DataFrame; delim = '\t'), CSV.read(DATAFILES[2], DataFrame; delim = '\t')) .|> process_data

    slopes_sp = vcat(map(x -> weighted_linear_regression(x, @formula(r1 ~ λ)), [df_sp]),
                     map(x -> weighted_linear_regression(x, @formula(r2 ~ λ)), [df_sp]))

    slopes_li = vcat(map(x -> weighted_linear_regression(x, @formula(r1 ~ λ)), [df_li]),
                     map(x -> weighted_linear_regression(x, @formula(r2 ~ λ)), [df_li]))

    println("Weighted linear regression for linear data:")
    println("Linear regression for spherical data (r1, r2):", slopes_sp)
    println("Linear regression for linear data (r1, r2):", slopes_li)

    println("\nFirst 5 rows of spherical data:")
    println(first(df_sp, 5))

    println("\nFirst 5 rows of linear data:")
    println(first(df_li, 5))

    formatted_df_sp = format_dataframe_for_output(df_sp)
    formatted_df_li = format_dataframe_for_output(df_li)

    CSV.write("./spherical-data.tsv", formatted_df_sp, delim = '\t')
    CSV.write("./linear-data.tsv", formatted_df_li, delim = '\t')
end

main()

