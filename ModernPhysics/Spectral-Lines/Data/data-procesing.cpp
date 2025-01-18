#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <string>
#include <tuple>

constexpr double kAlpha = 7.761e-37;
constexpr double kDParameter = 1.0e6 / 600.0;
constexpr double kDegToRad = M_PI / 180.0;
constexpr double kh = 4.135667e-15;
constexpr double kPlanckDivisor = 3.0;

double ComputeFranhoffer(double angle_deg) {
	return kDParameter * std::sin(angle_deg * kDegToRad);
}

double ComputeWavelengthUncertainty(double angle_deg, double angle_uncertainty) {
	return kDParameter * std::cos(angle_deg * kDegToRad) * angle_uncertainty * kDegToRad;
}

double ComputePlanckConstant(double wavelength, double quantum_number) {
	return std::exp(std::log(kAlpha * wavelength * (0.25 - 1.0 / (quantum_number * quantum_number))) / kPlanckDivisor);
}

double ComputePlanckUncertainty(double wavelength_uncertainty, double quantum_number, double planck_constant) {
	return kAlpha * (0.25 - 1.0 / (quantum_number * quantum_number)) * wavelength_uncertainty /
		(kPlanckDivisor * planck_constant * planck_constant);
}

std::tuple<double, double, double, double> ProcessLine(double quantum_number, double angle, double angle_uncertainty) {
	const double wavelength = ComputeFranhoffer(angle);
	const double wavelength_uncertainty = ComputeWavelengthUncertainty(angle, angle_uncertainty);
	const double planck_constant = ComputePlanckConstant(wavelength, quantum_number);
	const double planck_uncertainty = ComputePlanckUncertainty(wavelength_uncertainty, quantum_number, planck_constant);
	return {wavelength, wavelength_uncertainty, planck_constant, planck_uncertainty};
}

void WriteProcessedData(std::ofstream& output_file, std::ifstream& data_file) {
	output_file << "#Color\tn\tφ(σϕ) [deg]\tλ(σλ) [nm]\th(σh) [eVs]\n";

	double planck_constant_r = 0.0, planck_constant_r_u = 0.0;
	std::string colour;
	double quantum_number, angle, angle_uncertainty;

	std::string header_line;
	std::getline(data_file, header_line);

	while (data_file >> quantum_number >> colour >> angle >> angle_uncertainty) {
		const auto [wavelength, wavelength_uncertainty, planck_constant, planck_uncertainty] =
			ProcessLine(quantum_number, angle, angle_uncertainty);

		output_file << std::fixed << std::setprecision(0)
			<< colour << "\t" << quantum_number << "\t"
			<< std::setprecision(1) << angle << "("
			<< std::setprecision(0) << angle_uncertainty * 10 << ")\t"
			<< std::setprecision(0) << wavelength << "("
			<< std::setprecision(0) << wavelength_uncertainty << ")\t"
			<< std::setprecision(3) << planck_constant * 1e12 << "("
			<< std::setprecision(0) << planck_uncertainty * 1e15 << ")e-15\n";

		planck_constant_r += planck_constant;
		planck_constant_r_u += planck_uncertainty * planck_uncertainty * 1e-6;
	}

	const double h = planck_constant_r * 1e-3 / kPlanckDivisor;
	const double σ = std::sqrt(planck_constant_r_u) / kPlanckDivisor;
	const double Err = std::abs(kh - h) / kh;

	std::cout << "h = " << h << "\n"
		<< "σ = " << σ << "\n"
		<< "Err = " << Err << "\n";

	output_file << "\n# Planck Constant =" << h << "\n"
		<< "# Uncertainty =" << σ << "\n"
		<< "# Relative percentage error =" << Err;
}

void ProcessDataFile(const std::string& input_file) {
	std::ifstream data_file(input_file);
	if (!data_file) return;

	std::ofstream output_file("data.tsv");
	if (!output_file) return;

	WriteProcessedData(output_file, data_file);
	std::cout << "Data has been processed and saved to: data.tsv\n";
}

int main() {
	ProcessDataFile("./hydrogen.csv");
	return 0;
}

