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

weighted_linear_regression(df::DataFrame, formula) = coef(lm(formula, df, wts = 1.0 ./ (df.λσ .^ 2)))

format_dataframe_for_output(df::DataFrame) = DataFrame(
"#U (kV)" => df[!, "U (kV)"],
"r1 (mm)" => round.(df[!, "r1"], digits = 2),
"r2 (mm)" => round.(df[!, "r2"], digits = 2),
"λ (pm)" => round.(df[!, "λ"], digits = 1),
"λσ (pm)" => round.(df[!, "λσ"], digits = 1)
)

main() = begin
  df_sp, df_li = (CSV.read(DATAFILES[1], DataFrame; delim = '\t'), CSV.read(DATAFILES[2], DataFrame; delim = '\t')) .|> process_data

  coefs_sp = [weighted_linear_regression(df_sp, @formula(r1 ~ λ)), weighted_linear_regression(df_sp, @formula(r2 ~ λ))]
  slopes_sp = vcat(coefs_sp[1][2], coefs_sp[2][2])
  parameters_d_sp = 2.0 * 65 ./ slopes_sp

  coefs_li = [weighted_linear_regression(df_li, @formula(r1 ~ λ)), weighted_linear_regression(df_li, @formula(r2 ~ λ))]
  slopes_li = vcat(coefs_li[1][2], coefs_li[2][2])
  parameters_d_li = 135.0 ./ slopes_li

  println("Weighted linear regression for spherical data (r1, r2):", slopes_sp, parameters_d_sp)
  println("Weighted linear regression for linear data (r1, r2):", slopes_li, parameters_d_li)

  println("Coefficients for spherical data:")
  println(coefs_sp)

  println("Coefficients for linear data:")
  println(coefs_li)

  formatted_df_sp = format_dataframe_for_output(df_sp)
  formatted_df_li = format_dataframe_for_output(df_li)

  println("\nFirst 5 rows of spherical data:")
  println(first(formatted_df_sp, 5))
  println("\nFirst 5 rows of linear data:")
  println(first(formatted_df_li, 5))

  CSV.write("./spherical-data.tsv", formatted_df_sp, delim = '\t')
  CSV.write("./linear-data.tsv", formatted_df_li, delim = '\t')

  parameters_df = DataFrame(
                            "Type" => ["Spherical r1", "Spherical r2", "Linear r1", "Linear r2"],
                            "d (mm)" => vcat(parameters_d_sp, parameters_d_li)
                           )
  CSV.write("./parameters-d.tsv", parameters_df, delim = '\t')
end

main()
