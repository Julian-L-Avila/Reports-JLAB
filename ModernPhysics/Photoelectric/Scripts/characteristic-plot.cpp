#include <gnuplot-iostream/gnuplot-iostream.h>
#include <array>

constexpr std::array<const char*, 3> kDataFiles = {"../Data/violet.csv", "../Data/blue.csv", "../Data/stop-potential.csv"};
constexpr char kOutputFile[] = "characteristics.tex";

void ConfigureGnuplot(Gnuplot& gp) {
  gp << "set term tikz standalone header '\\usepackage{siunitx}'\n"
    << "set output '" << kOutputFile << "'\n"
    << "set grid\n"
    << "set key L l t\n"
    << "set style data linespoints\n"
    << "set xlabel '$U \\, [\\si{\\volt}]$'\n"
    << "set ylabel '$I_p \\, [\\si{\\nano\\ampere}]$'\n";
}

void PlotData(Gnuplot& gp, const char* file, const char* title) {
  gp << "set title '" << title << "'\n"
    << "plot for [i=1:3] '" << file
    << "' using ($1==i ? $2 : 1/0):($1==i ? $3 : 1/0) title sprintf('$I_{%d}$', i)\n";
}

void LinearFit(Gnuplot& gp, const char* file) {
  gp << "K(v) = h * v * 1.0e4 / 2.998 + b\n"
    << "fit K(x) '" << file << "' u 1:2 via h, b\n"
    << "set xlabel '$\\nu \\, [\\si{\\tera\\Hz}]$'\n"
    << "set ylabel '$K_{m} \\, [\\si{\\eV}]$'\n"
    << "set title 'Energía Cinética en Función de Frecuencia'\n"
    << "set label sprintf('$h = \\qty{%.3e}{\\eV}$', h * 1.0e-12) at graph 0.1,0.7\n"
    << "plot '" << file << "' u 1:2 title 'Data' with points, K(x) with lines title 'Fit'\n"
    << "print('h = ', h * 1.0e-12)\n";
}

int main() {
  Gnuplot gp;
  ConfigureGnuplot(gp);
  PlotData(gp, kDataFiles[0], "Curva Característica usando Filtro Violeta");
  PlotData(gp, kDataFiles[1], "Curva Característica usando Filtro Azul");
  LinearFit(gp, kDataFiles[2]);
}

