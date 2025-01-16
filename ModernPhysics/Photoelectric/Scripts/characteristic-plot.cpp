#include <gnuplot-iostream/gnuplot-iostream.h>
#include <array>

constexpr std::array<const char*, 3> kDataFiles = {"../Data/violet.csv", "../Data/blue.csv", "../Data/stop-potential.csv"};
constexpr char kOutputFile[] = "characteristics.tex";

void ConfigureGnuplot(Gnuplot& gp) {
  gp << "set term tikz standalone header '\\usepackage{siunitx}'\n"
    << "set output '" << kOutputFile << "'\n"
    << "set grid\nset key L l t\nset style data linespoints\n"
    << "set xlabel '$V \\, [\\si{\\volt}]$'\n"
    << "set ylabel '$I_p \\, [\\si{\\nano\\ampere}]$'\n";
}

void PlotData(Gnuplot& gp, const char* file, const char* title, int i) {
  gp << "set title '" << title << "'\n"
    << "plot for [j=3:1:-" << i << "] '" << file
    << "' using ($1==j ? $2 : 1/0):($1==j ? $3 : 1/0) lw 2 title sprintf('$I_{%d}$', j)\n";
}

void LinearFit(Gnuplot& gp, const char* file) {
  gp << "K(v) = h * v + b\n"
    << "fit K(x) '" << file << "' u 1:2:(0.001) yerror via h, b\n"
    << "set xlabel '$\\nu \\, [\\si{\\tera\\Hz}]$'\n"
    << "set ylabel '$K_{m} \\, [\\si{\\eV}]$'\n"
    << "set xrange [550:750]\n"
    << "set title 'Energía Cinética en Función de Frecuencia'\n"
    << "set label sprintf('$h = \\qty{%.3e}{\\eV\\s}$', h * 1e-12) at graph 0.1,0.6\n"
    << "set label sprintf('$\\phi = \\qty{%.3e}{\\eV}$', -b) at graph 0.1,0.5\n"
    << "plot '" << file << "' u 1:2 title 'Data' with points, K(x) with lines lw 2 title 'Fit'\n"
    << "print('h = ', h * 1.0e-12)\n";
}

int main() {
  Gnuplot gp;
  ConfigureGnuplot(gp);
  PlotData(gp, kDataFiles[0], "Curva Característica usando Filtro Violeta", 2);
  PlotData(gp, kDataFiles[1], "Curva Característica usando Filtro Azul", 1);
  LinearFit(gp, kDataFiles[2]);
}

