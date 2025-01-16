#include <gnuplot-iostream/gnuplot-iostream.h>
#include <array>

constexpr std::array<const char*, 3> kDataFiles = {"../Data/violet.csv", "../Data/blue.csv", "../Data/stop-potential.csv"};
constexpr char kOutputFile[] = "characteristics.tex";

void ConfigureGnuplot(Gnuplot& gp) {
  gp << "set term tikz standalone header '\\usepackage{siunitx}'\n"
    << "set output '" << kOutputFile << "'\n"
    << "set grid\nset key L l t\nset style data linespoints\n"
    << "set xlabel '$V \\, [\\si{\\volt}]$'\n"
    << "set ylabel '$I_p \\, [\\si{\\nano\\ampere}]$'\n"
    << "set xtics 3\nset mxtics 3\nset mytics 5\n";
}

void PlotData(Gnuplot& gp, const char* file, const char* title, int i, int uppery) {
  gp << "set title '" << title << "'\n"
    << "set yrange [-" << uppery / 20 << ":" << uppery << "]\n"
    << "set ytics " << uppery / 4 << "\n"
    << "plot for [j=3:1:-" << i << "] '" << file
    << "' using ($1==j ? $2 : 1/0):($1==j ? $3 : 1/0) lw 2 ps 2 title sprintf('$I_{%d}$', j)\n";
}

void LinearFit(Gnuplot& gp, const char* file) {
  gp << "K(v) = h * v + b\n"
    << "fit K(x) '" << file << "' u 2:3:(0.001) yerror via h, b\n"
    << "set xlabel '$\\nu \\, [\\si{\\tera\\Hz}]$'\n"
    << "set ylabel '$K_{m} \\, [\\si{\\eV}]$'\n"
    << "set xrange [650:850]\n"
    << "set yrange [0.7:1.6]\n"
    << "set xtics 50\nset mxtics 5\nset ytics 0.2\nset mytics 2\n"
    << "set title 'Energía Cinética en Función de Frecuencia'\n"
    << "set label sprintf('$h = \\qty{%.3e}{\\eV\\s}$', h * 1e-12) at graph 0.1,0.6\n"
    << "set label sprintf('$\\phi = \\qty{%.3e}{\\eV}$', -b) at graph 0.1,0.5\n"
    << "plot '" << file << "' u 2:3 title 'Data' with points ps 2, K(x) with lines lw 2 title 'Fit'\n"
    << "print('h = ', h * 1.0e-12)\n"
    << "print('v_0 = ', -b / h)\n"
    << "print('λ_0 = ', -h * 2.998e5 / b)\n";
}

int main() {
  Gnuplot gp;
  ConfigureGnuplot(gp);
  for (int i = 0; i < 2; ++i) PlotData(gp, kDataFiles[i], i ? "Curva Característica usando Filtro Azul" : "Curva Característica usando Filtro Ultravioleta", 2 - i, i ? 360 : 140);
  LinearFit(gp, kDataFiles[2]);
}

