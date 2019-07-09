#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_odeiv2.h>

/**************************** constants ************************/
#define M_PI 3.14159265358979323846

#define T_eq 0.691848671251
#define h 0.673

struct Parameter {
    double M_pl;
    double Lambda_QCD;
    double T0;
    double rho_c;
    double m_u;
    double m_d;
    double m_pi0;
    double f_pi0;
};

/************************** data *****************************/
struct Data {
    struct Parameter parameter;
    gsl_interp *g_s_interp;
    gsl_interp *g_rho_interp;
    gsl_interp *m_a_interp;
    gsl_interp_accel *g_s_accel;
    gsl_interp_accel *g_rho_accel;
    gsl_interp_accel *m_a_accel;
    double f_a;
};

/**************************** g_star **************************/
double g_s_T_data[] = {
    1.000000000000000000e+06,
    3.162277660168379545e+06,
    8.889310880618948489e+06,
    1.000000000000000000e+07,
    1.357619821449235640e+07,
    1.701623770106690377e+07,
    2.044362516790480167e+07,
    2.673217469914484024e+07,
    3.495510988273700327e+07,
    3.981071705534973741e+07,
    4.570745630183636397e+07,
    5.414409298477500677e+07,
    5.976727204099109024e+07,
    6.691232499567509443e+07,
    8.626848390004968643e+07,
    1.000000000000000000e+08,
    1.021918345902832150e+08,
    1.176845779833682477e+08,
    1.299068060728070587e+08,
    1.374526915891980231e+08,
    1.412537544622753859e+08,
    1.496012642413134575e+08,
    1.582911400622894466e+08,
    1.584893192461114228e+08,
    1.674857839556941688e+08,
    1.772145163413116932e+08,
    1.875083607716387212e+08,
    2.012205428893491030e+08,
    2.190051519510276914e+08,
    2.451867619626995325e+08,
    2.511886431509579718e+08,
    2.522073215230326355e+08,
    2.706508336411698461e+08,
    2.904430898687081933e+08,
    3.161135146365276575e+08,
    3.162277660168379545e+08,
    3.392303829443078637e+08,
    3.640377503152915239e+08,
    4.075577055463561416e+08,
    4.562803808295952678e+08,
    5.180895325948365331e+08,
    5.638801843038603067e+08,
    6.312908865053075552e+08,
    6.968541177958354950e+08,
    7.584446387240841389e+08,
    8.491151164257112741e+08,
    9.241630224553511143e+08,
    1.000000000000000000e+09,
    1.034644787955297947e+09,
    1.158334418530343056e+09,
    1.296810886955338478e+09,
    1.472480901030686617e+09,
    1.625406609208719015e+09,
    1.794214541872909307e+09,
    2.095598440296609640e+09,
    2.413300622869539261e+09,
    2.779167890351248264e+09,
    3.155642569370943069e+09,
    3.685712880023995399e+09,
    4.491032543701240540e+09,
    5.472312674679186821e+09,
    6.040644182789868355e+09,
    6.858928528864598274e+09,
    7.678899156225990295e+09,
    8.843054929001392365e+09,
    1.000000000000000000e+10,
    1.004096250324603462e+10,
    1.092841956980114746e+10,
    1.206339952153341103e+10,
    1.331625374434455299e+10,
    1.599839547216831779e+10,
    1.842382443863255692e+10,
    1.995262314968878937e+10,
    2.121695938421191025e+10,
    2.409107433123492050e+10,
    2.622033177535417938e+10,
    2.977221902303988266e+10,
    3.477321930727656555e+10,
    3.784660387485523987e+10,
    3.981071705534969330e+10,
    4.119162543459052277e+10,
    4.546961538079605103e+10,
    4.948838663606021881e+10,
    5.462804350197291565e+10,
    5.862290366794455719e+10,
    6.380421074934008789e+10,
    6.847010913419790649e+10,
    7.452174832509204102e+10,
    7.997140283947570801e+10,
    8.703956852073117065e+10,
    9.744496774669789124e+10,
    1.000000000000000000e+11,
    1.138133248053891754e+11,
    1.386812706920658875e+11,
    1.619763120731806641e+11,
    1.891843472582878418e+11,
    2.178655482349816284e+11,
    2.508949487397389832e+11,
    2.818382931264454956e+11
};

double g_s_data[] = {
    1.068563674821407261e+01,
    1.073688630297213997e+01,
    1.069075394693280145e+01,
    1.075483767791460110e+01,
    1.089898514333540902e+01,
    1.111381189799510594e+01,
    1.133005198656704238e+01,
    1.176582994283950256e+01,
    1.286870150568677218e+01,
    1.339089067042551306e+01,
    1.374920853300912427e+01,
    1.441064880393506087e+01,
    1.462971556033147635e+01,
    1.529304027647370390e+01,
    1.639638295062511020e+01,
    1.721003870059810126e+01,
    1.728018775707602117e+01,
    1.838729932165998093e+01,
    2.016291782673113175e+01,
    2.193994966571446525e+01,
    2.283182986634795242e+01,
    2.505022649523940004e+01,
    2.704962286974773633e+01,
    2.773801334845414601e+01,
    2.993847738635588485e+01,
    3.216023829638916709e+01,
    3.504909281299728718e+01,
    3.793747621830137007e+01,
    4.038065944125142437e+01,
    4.393472311921813400e+01,
    4.507246649955709472e+01,
    4.504560357423476091e+01,
    4.704452883743900315e+01,
    4.859872502959332508e+01,
    5.015245011044356716e+01,
    5.066386474352851366e+01,
    5.170664630259787486e+01,
    5.348320703027714274e+01,
    5.592544803061905156e+01,
    5.792295995991105428e+01,
    6.080945891999880359e+01,
    6.280791307189896600e+01,
    6.458306046566600855e+01,
    6.613631443521215658e+01,
    6.769003951606241287e+01,
    6.946518690982944122e+01,
    7.079654745515472314e+01,
    7.219634891626874662e+01,
    7.234933031339679133e+01,
    7.390211317163888793e+01,
    7.523253149435601017e+01,
    7.634011417024409241e+01,
    7.700390999769041400e+01,
    7.789007036066170997e+01,
    7.899671081394163252e+01,
    7.943672877195076865e+01,
    8.009911126548486493e+01,
    8.098432940584798700e+01,
    8.120151171702808313e+01,
    8.163964522982091410e+01,
    8.185541420708877070e+01,
    8.185211642796022602e+01,
    8.207024096174848182e+01,
    8.228883660684080326e+01,
    8.250649002932497922e+01,
    8.299791256754191693e+01,
    8.272461456311323502e+01,
    8.316651696633869051e+01,
    8.294085465168517146e+01,
    8.315992140808158695e+01,
    8.404325510322840387e+01,
    8.448327306123753999e+01,
    8.522846128559902468e+01,
    8.514565555477162206e+01,
    8.603087369513475835e+01,
    8.691750516941011995e+01,
    8.802508784529820218e+01,
    8.935409283410308490e+01,
    9.024072430837846071e+01,
    9.116139839622547925e+01,
    9.112735578265382230e+01,
    9.245824521667502438e+01,
    9.334487669095038598e+01,
    9.445340158944662790e+01,
    9.556286871055102949e+01,
    9.644950018482639109e+01,
    9.689187369935592642e+01,
    9.800086970915621976e+01,
    9.888797229473568962e+01,
    9.977460376901103700e+01,
    1.006602930206782531e+02,
    1.014094292803970205e+02,
    1.017669334739581757e+02,
    1.028721605933258729e+02,
    1.035340719755558894e+02,
    1.039736188222609456e+02,
    1.041912722447451216e+02,
    1.044089256672292834e+02,
    1.049558601521650161e+02
};

double g_rho_T_data[] = {
    1.000000000000000000e+06,
    3.162277660168379545e+06,
    1.000000000000000000e+07,
    1.088453338955713809e+07,
    1.326475968926919624e+07,
    1.527738698708757758e+07,
    1.778279410038922727e+07,
    1.970055801287817582e+07,
    2.333985531532329693e+07,
    2.688115052421820164e+07,
    3.095975719400755316e+07,
    3.616447864454406500e+07,
    3.981071705534973741e+07,
    4.469992026717726886e+07,
    5.148212227064099908e+07,
    5.764162242789599299e+07,
    6.545621976095777750e+07,
    7.225962535056763887e+07,
    8.806133723945532739e+07,
    1.000000000000000000e+08,
    1.073185624583393633e+08,
    1.218679681941691041e+08,
    1.345347127607586980e+08,
    1.412537544622753859e+08,
    1.443807197423019707e+08,
    1.527738698708760142e+08,
    1.584893192461114228e+08,
    1.616549311915154755e+08,
    1.734857482977517247e+08,
    1.942421865872219801e+08,
    2.114235565569203496e+08,
    2.268967215951809883e+08,
    2.469664942183795273e+08,
    2.511886431509579718e+08,
    2.688115052421818972e+08,
    2.967513053711061478e+08,
    3.162277660168379545e+08,
    3.275951197108094692e+08,
    3.616447864454412460e+08,
    4.165162041238636971e+08,
    4.345470538341261744e+08,
    4.797131185075845718e+08,
    5.295736653552706838e+08,
    5.929337004735993147e+08,
    6.733189831779426336e+08,
    7.433025867296336889e+08,
    8.322338933253636360e+08,
    9.450616120697368383e+08,
    1.000000000000000000e+09,
    1.043289671642403841e+09,
    1.201585285122544050e+09,
    1.423554941207698345e+09,
    1.710522669918651819e+09,
    1.970055801287812471e+09,
    2.268967215951805592e+09,
    2.613231677853355408e+09,
    3.095975719400753975e+09,
    3.616447864454404831e+09,
    4.165162041238636494e+09,
    4.865377746586316109e+09,
    5.683308448172122002e+09,
    6.453806660598555565e+09,
    7.328762954222970009e+09,
    8.931414506078479767e+09,
    1.000000000000000000e+10,
    1.073185624583393669e+10,
    1.253601517565840721e+10,
    1.464347573098955154e+10,
    1.809958893761981201e+10,
    1.995262314968878937e+10,
    2.301246760682714462e+10,
    2.804482670378157043e+10,
    3.229999491706467438e+10,
    3.981071705534969330e+10,
    4.049132264540113831e+10,
    5.148212227064096832e+10,
    6.545621976095760345e+10,
    7.977016507938925171e+10,
    9.450616120697348022e+10,
    1.000000000000000000e+11,
    1.119643477882954559e+11,
    1.464347573098955383e+11,
    1.734857482977517090e+11,
    2.144313814840795898e+11,
    2.576575926597682800e+11,
    2.818382931264454956e+11
};

double g_rho_data[] = {
    1.071000000000000085e+01,
    1.074000000000000021e+01,
    1.075999999999999979e+01,
    1.080125135131972947e+01,
    1.090552145973249587e+01,
    1.103397081912347844e+01,
    1.108999999999999986e+01,
    1.137203588584481473e+01,
    1.164646262389609710e+01,
    1.195896645156457971e+01,
    1.253768379852160564e+01,
    1.328947911783417624e+01,
    1.367999999999999972e+01,
    1.397679100210582881e+01,
    1.453320388923406981e+01,
    1.485016962583769562e+01,
    1.547152468595485963e+01,
    1.595679736457547193e+01,
    1.685719566897109090e+01,
    1.760999999999999943e+01,
    1.800484815831046603e+01,
    1.952499495948487152e+01,
    2.220666589561509241e+01,
    2.407000000000000028e+01,
    2.504256604295846600e+01,
    2.733674510268912528e+01,
    2.983999999999999986e+01,
    3.129752612925675948e+01,
    3.394838142103937884e+01,
    3.962507282485661619e+01,
    4.242269097400248512e+01,
    4.433881935504551564e+01,
    4.708260167064418766e+01,
    4.782999999999999829e+01,
    4.943500559698222219e+01,
    5.133328247827917323e+01,
    5.303999999999999915e+01,
    5.374277105177441172e+01,
    5.549273512266059782e+01,
    5.841589989986223230e+01,
    5.902703475281153089e+01,
    6.082963192501539851e+01,
    6.309879926514339843e+01,
    6.531919337049082230e+01,
    6.722029195039360161e+01,
    6.877466885127574869e+01,
    7.063494924587689638e+01,
    7.258337136300937686e+01,
    7.348000000000000398e+01,
    7.361383182648822299e+01,
    7.550980309826840653e+01,
    7.715829396508244997e+01,
    7.837513627197488120e+01,
    7.937013314966557687e+01,
    7.990452907673306981e+01,
    8.033849902440101687e+01,
    8.138296437891013113e+01,
    8.154996486431866742e+01,
    8.176313530904720039e+01,
    8.203797569515631949e+01,
    8.213799549737022687e+01,
    8.212096942396867405e+01,
    8.238747776382557220e+01,
    8.282214566179149529e+01,
    8.309999999999999432e+01,
    8.330167584821614923e+01,
    8.317738244742008646e+01,
    8.378515591178110355e+01,
    8.470143484900721376e+01,
    8.556000000000000227e+01,
    8.610937974610982337e+01,
    8.814524705691688666e+01,
    8.944617549335012541e+01,
    9.196999999999999886e+01,
    9.201896679249426825e+01,
    9.466386195413484472e+01,
    9.754281562017601459e+01,
    9.981079594100508245e+01,
    1.011706701909447474e+02,
    1.021700000000000017e+02,
    1.024537372446345813e+02,
    1.037503043134525598e+02,
    1.042569268893767571e+02,
    1.044228085230176788e+02,
    1.047461089798758564e+02,
    1.049800000000000040e+02
};

double a0[] = {1.21, 1.36};

double a[2][3][5] = {
    {{0.572, 0.33, 0.579, 0.138, 0.108},
     {-8.77, -2.95, -1.8, -0.162, 3.76},
     {0.682, 1.01, 0.165, 0.934, 0.869}},
    {{0.498, 0.327, 0.579, 0.14, 0.109},
     {-8.74, -2.89, -1.79, -0.102, 3.82},
     {0.693, 1.01, 0.155, 0.963, 0.907}}
};

double g_i_shellard(int i, double T) {
    double t = log(T / 1e9);
    double ans = 0.0;
    for(int k = 0; k < 5; k++) {
        ans += a[i][0][k] * (1.0 + tanh((t - a[i][1][k]) / a[i][2][k]));
    }
    return exp(a0[i] + ans);
}

double sech(double x) { return 1 / cosh(x); }

double dgdT_i_shellard(int i, double T) {
    double t = log(T / 1e9);
    double x = 0.0;
    for(int k = 0; k < 5; k++) {
        x += (t - a[i][1][k]) / a[i][2][k];
    }
    x = sech(x);
    x *= x;
    double ans = 0.0;
    for(int k = 0; k < 5; k++) {
        ans += a[i][0][k] / a[i][2][k] * x / T;
    }
    return ans * g_i_shellard(i, T);
}


double d2dT2_g_i(int i, double T) {
    double t = log(T / 1e9);
    double x = 0.0;
    for(int k = 0; k < 5; k++) {
        x += (t - a[i][1][k]) / a[i][2][k];
    }
    double y = sech(x);
    y *= y;
    double ans = 0;
    for(int k = 0; k < 5; k++) {
        ans += a[i][0][k] / a[i][2][k] * y / (T*T) * g_i_shellard(i, T) * (
            -2 * tanh(x) / a[i][2][k] + a[i][0][k] / a[i][2][k] * y - 1
        );
    }
    return ans;
}

#define T_min 1e6

double g_s_bosamyi(struct Data *data, double T) {
    return gsl_interp_eval(data->g_s_interp, g_s_T_data, g_s_data, T, data->g_s_accel);
}

double g_rho_bosamyi(struct Data *data, double T) {
    return gsl_interp_eval(data->g_rho_interp, g_rho_T_data, g_rho_data, T, data->g_rho_accel);
}

double dg_s_bosamyi(struct Data *data, double T) {
    return gsl_interp_eval_deriv(data->g_s_interp, g_s_T_data, g_s_data, T, data->g_s_accel);
}

double dg_rho_bosamyi(struct Data *data, double T) {
    return gsl_interp_eval_deriv(data->g_rho_interp, g_rho_T_data, g_rho_data, T, data->g_rho_accel);
}

double ddg_rho_bosamyi(struct Data *data, double T) {
    return gsl_interp_eval_deriv2(data->g_rho_interp, g_rho_T_data, g_rho_data, T, data->g_rho_accel);
}

/***************************** axion mass **********************/
double m_a_T_data[] = {
    1.000000000000000000e+08,
    1.200000000000000000e+08,
    1.400000000000000000e+08,
    1.700000000000000000e+08,
    2.000000000000000000e+08,
    2.400000000000000000e+08,
    2.900000000000000000e+08,
    3.500000000000000000e+08,
    4.200000000000000000e+08,
    5.000000000000000000e+08,
    6.000000000000000000e+08,
    7.200000000000000000e+08,
    8.600000000000000000e+08,
    1.000000000000000000e+09,
    1.200000000000000000e+09,
    1.500000000000000000e+09,
    1.800000000000000000e+09,
    2.100000000000000000e+09,
    2.500000000000000000e+09,
    3.000000000000000000e+09
};

double m_a_data[] = {
    3.317003017060090594e+31,
    3.394265942731709143e+31,
    2.696161274474827632e+31,
    1.001718896794929358e+31,
    2.888988855633805381e+30,
    6.176548650339513672e+29,
    1.176917082581084981e+29,
    2.758962938228836627e+28,
    6.930201569574824287e+27,
    1.908736008284217085e+27,
    4.906206986112828193e+26,
    1.261089374649555551e+26,
    3.241498810293842533e+25,
    1.025051927325459805e+25,
    2.458928306972691242e+24,
    4.175859389174863930e+23,
    9.566341197167313342e+22,
    2.696161274474827402e+22,
    6.320418950373564219e+21,
    1.351281610567017300e+21
};

#define callibration_factor 3.44402370315
#define C 0.018
#define n_fox 4.0
#define d 1.2

double m_a(struct Data* data, double T) {
    const int N_m_a_data = sizeof(m_a_T_data) / sizeof(double);
    const double m_a0 = data->parameter.m_pi0 * data->parameter.f_pi0 * sqrt(data->parameter.m_u * data->parameter.m_d) / (data->parameter.m_u + data->parameter.m_d) / data->f_a;
    if(T < m_a_T_data[0]) {
        return m_a0;
    } else if(T > m_a_T_data[N_m_a_data - 1]) {
        return callibration_factor * C * m_a0 * pow(data->parameter.Lambda_QCD / T, n_fox) * pow(1 - log(data->parameter.Lambda_QCD / T), d);
    } else {
        return sqrt(gsl_interp_eval(data->m_a_interp, m_a_T_data, m_a_data, T, data->m_a_accel)) / data->f_a;
    }
}

/*************************** rhs function **************************/
#define temperature_unit 1e12

int rhs(double _T, double y[], double dydT[], void* _params) {
    struct Data* data = (struct Data*) _params;
    double theta = y[0];
    double dthetadT = y[1];

    double T = temperature_unit * _T;

    // compute g quantaties
    double g_s = T < T_min ?
        g_s_bosamyi(data, T_min) / g_i_shellard(1, T_min) * g_i_shellard(1, T) :
        g_s_bosamyi(data, T);
    double g_rho = T < T_min ?
        g_rho_bosamyi(data, T_min) / g_i_shellard(0, T_min) * g_i_shellard(0, T) :
        g_rho_bosamyi(data, T);
    double dg_s = T < T_min ?
        dg_s_bosamyi(data, T_min) / dgdT_i_shellard(1, T_min) * dgdT_i_shellard(1, T) :
        dg_s_bosamyi(data, T);
    double dg_rho = T < T_min ?
        dg_rho_bosamyi(data, T_min) / dgdT_i_shellard(0, T_min) * dgdT_i_shellard(0, T) :
        dg_rho_bosamyi(data, T);
    double ddg_rho = T < T_min ?
        ddg_rho_bosamyi(data, T_min) / d2dT2_g_i(0, T_min) * d2dT2_g_i(0, T) :
        ddg_rho_bosamyi(data, T);

    double rho = M_PI*M_PI / 30.0 * g_rho * pow(T, 4);
    double H = sqrt(rho / (3 * data->parameter.M_pl * data->parameter.M_pl));

    double dtdT = - sqrt(8 * M_PI) * data->parameter.M_pl * sqrt(45 / (64 * pow(M_PI, 3))) *
        1 / (pow(T, 3) * g_s * sqrt(g_rho)) *
        (T * dg_rho + 4 * g_rho);
    double d2tdT2 = - sqrt(8 * M_PI) * data->parameter.M_pl * sqrt(45 / (64 * pow(M_PI, 3))) * (
            - (3 * T * T * g_s * sqrt(g_rho) + pow(T, 3) * dg_s * sqrt(g_rho) + pow(T, 3) * g_s * dg_rho / (2 * sqrt(g_rho)))
              / pow(pow(T, 3) * g_s * sqrt(g_rho), 2)
              * (T * dg_rho + 4 * g_rho)
            + (dg_rho + T * ddg_rho + 4 * dg_rho) /
              (pow(T, 3) * g_s * sqrt(g_rho))
    );
    dtdT *= temperature_unit;
    d2tdT2 *= temperature_unit * temperature_unit;
    double m = m_a(data, T);

    double dVdtheta = sin(theta);

    double d2thetadT2 = -(3 * H * dtdT - d2tdT2 / dtdT) * dthetadT - m*m * dtdT*dtdT * dVdtheta;

    dydT[0] = dthetadT;
    dydT[1] = d2thetadT2;

    return GSL_SUCCESS;
}

/**************************** solver algorithm **********************/

int sign(double x) {
    if(x < 0.0) return -1;
    if(x > 0.0) return 1;
    return 0;
}

double simpson(double y[], double delta_x, int n) {
    double ans = 0.0;
    for(int i = 1; i <= n / 2; i++) {
        ans += y[2*i - 2] + 4 * y[2*i - 1] + y[2*i];
    }
    ans *= delta_x / 3.0;
    if(n % 2 == 1) {
        ans += delta_x * (y[n - 2] + y[n - 1]) / 2;
    }
    return ans;
}

double solver(struct Parameter parameter, double T_osc, double theta_i, double f_a) {
    struct Data data;
    data.parameter = parameter;

    const gsl_interp_type* my_interp_method = gsl_interp_steffen;

    const size_t N_g_s = sizeof(g_s_T_data) / sizeof(double);
    data.g_s_interp = gsl_interp_alloc(my_interp_method, N_g_s);
    gsl_interp_init(data.g_s_interp, g_s_T_data, g_s_data, N_g_s);
    data.g_s_accel = gsl_interp_accel_alloc();

    const size_t N_g_rho = sizeof(g_rho_T_data) / sizeof(double);
    data.g_rho_interp = gsl_interp_alloc(my_interp_method, N_g_rho);
    gsl_interp_init(data.g_rho_interp, g_rho_T_data, g_rho_data, N_g_rho);
    data.g_rho_accel = gsl_interp_accel_alloc();

    const size_t N_m_a = sizeof(m_a_T_data) / sizeof(double);
    data.m_a_interp = gsl_interp_alloc(my_interp_method, N_m_a);
    gsl_interp_init(data.m_a_interp, m_a_T_data, m_a_data, N_m_a);
    data.m_a_accel = gsl_interp_accel_alloc();

    data.f_a = f_a;

    gsl_odeiv2_system sys;
    sys.function = rhs;
    sys.jacobian = NULL;
    sys.dimension = 2;
    sys.params = (void*) &data;

    gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, -1e-6, 1e-6, 1e-6);

    // gsl_root_fsolver* fsolver = gsl_root_fsolver_alloc(
    // gsl_function F;
    // F.function = f;
    // F.params = data;
    // gsl_root_fsolver_set(fsolver, F,
    // double T_osc = ; // TODO:
    // gsl_root_fsolver_free(fsolver);

    T_osc /= temperature_unit;
    double T_start = 5 * T_osc;
    const int N = 300;
    const double avg_start = 0.8;
    const double avg_stop = 0.6;
    const double eps = 1e-5;
    const int num_crossings = 3;
    double y[] = {theta_i, 0.0};
    double next_T = T_start;
    double T = T_start;
    double delta_T = (T_osc - T_start) / N; // delta_T is negative

    while(y[0] > 0.0) {
        next_T += delta_T;
        int status = gsl_odeiv2_driver_apply(driver, &T, next_T, y);
        if(status != GSL_SUCCESS) {
            fprintf(stderr, "invalid integration");
            exit(-1);
        }
    }

    double T_s = T;
    double T_inteval = (avg_stop - avg_start) * T_s;
    delta_T = T_inteval / N;
    double last_n_over_s = 0.0 / 0.0; // NAN
    double T_vals[N], theta_vals[N], dthetadT_vals[N];

    while(1) {
        int zero_crossings = 0;
        int prev_sign = sign(y[0]);

        for(int i = 0; i < N; i++) {
            double next_T = T + delta_T;
            int status = gsl_odeiv2_driver_apply(driver, &T, next_T, y);
            if(status != GSL_SUCCESS) {
                exit(-1);
            }
            int s = sign(y[0]);
            if(prev_sign != s) zero_crossings++;
            prev_sign = s;
            T_vals[i] = T * temperature_unit;
            theta_vals[i] = y[0];
            dthetadT_vals[i] = y[1] / temperature_unit;
        }
        // integrate n/s

        double n_over_s[N];
        for(int i = 0; i < N; i++) {
            double T = T_vals[i];
            double g_s = T < T_min ?
                g_s_bosamyi(&data, T_min) / g_i_shellard(1, T_min) * g_i_shellard(1, T) :
                g_s_bosamyi(&data, T);
            double g_rho = T < T_min ?
                g_rho_bosamyi(&data, T_min) / g_i_shellard(0, T_min) * g_i_shellard(0, T) :
                g_rho_bosamyi(&data, T);
            double dg_rho = T < T_min ?
                dg_rho_bosamyi(&data, T_min) / dgdT_i_shellard(0, T_min) * dgdT_i_shellard(0, T) :
                dg_rho_bosamyi(&data, T);
            double m = m_a(&data, T);
            double V = 1 - cos(theta_vals[i]);
            double dtdT = - sqrt(8 * M_PI) * data.parameter.M_pl * sqrt(45 / (64 * pow(M_PI, 3))) *
                1 / (pow(T, 3) * g_s * sqrt(g_rho)) *
                (T * dg_rho + 4 * g_rho);
            n_over_s[i] = 45 / (2 * M_PI * M_PI) * data.f_a * data.f_a / (m * g_s * pow(T, 3)) *
                (0.5 * pow(dthetadT_vals[i] / dtdT, 2) + m*m * V);
        }
        if(T * temperature_unit < T_eq) {
            fprintf(stderr, "over T_eq\n");
            return 0.0 / 0.0;
        }
        double n_over_s_avg = simpson(n_over_s, T_vals[1] - T_vals[0], N) / (temperature_unit * T_inteval);

        double d_n_over_s_dT = (last_n_over_s - n_over_s_avg) / (temperature_unit * T_inteval);

        if(fabs(d_n_over_s_dT) < eps && zero_crossings > num_crossings) {
            double s_today = 2.0 * M_PI * M_PI / 45.0 * 43.0 / 11.0 * pow(data.parameter.T0, 3);
            double n_a_today = n_over_s_avg * s_today;
            double rho_a_today = m_a(&data, data.parameter.T0) * n_a_today;
            double Omega_a_h_sq_today = h*h * rho_a_today / data.parameter.rho_c;
            return Omega_a_h_sq_today;
        }

        last_n_over_s = n_over_s_avg;

    }
}
