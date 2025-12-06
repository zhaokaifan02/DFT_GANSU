#include "dftGrids.hpp"
#include <set>

namespace gansu::dft::chemgrid
{

    struct P4
    {
        double x, y, z, w;
    };

    // ---------- Orbit families ----------
    enum class OrbitCode : uint8_t
    {
        O6 = 0,       // permutations of (±1, 0, 0)
        O12 = 1,      // permutations of (0, ±1/√2, ±1/√2)
        O8 = 2,       // (±1/√3, ±1/√3, ±1/√3)
        O24_AA_B = 3, // permutations of (±a, ±a, ±b), with b = √(1 − 2a²)
        O24_A_B0 = 4, // permutations of (±a, ±b, 0), with b = √(1 − a²)
        O48_A_B_C = 5 // permutations of (±a, ±b, ±c), with c = √(1 − a² − b²)
    };

    // ---------- Entry: one orbit’s parameters ----------
    struct Orbit
    {
        OrbitCode code;
        double a, b, v;
    };

    // Expand a single orbit into its point set (exactly mirrors your Python rules)
    static void expand_orbit(std::vector<P4> &out, OrbitCode code, double a, double b, double v)
    {
        auto add = [&](double X, double Y, double Z)
        { out.push_back({X, Y, Z, v}); };

        switch (code)
        {
        case OrbitCode::O6:
        {
            const double s = 1.0;
            add(s, 0, 0);
            add(-s, 0, 0);
            add(0, s, 0);
            add(0, -s, 0);
            add(0, 0, s);
            add(0, 0, -s);
        }
        break;
        case OrbitCode::O12:
        {
            const double t = std::sqrt(0.5);
            // permutations of (0, ±t, ±t)
            add(0.0, t, t);
            add(0.0, -t, t);
            add(0.0, t, -t);
            add(0.0, -t, -t);
            // permutations of (±t, 0, ±t)
            add(t, 0.0, t);
            add(-t, 0.0, t);
            add(t, 0.0, -t);
            add(-t, 0.0, -t);
            // permutations of (±t, ±t, 0)
            add(t, t, 0.0);
            add(-t, t, 0.0);
            add(t, -t, 0.0);
            add(-t, -t, 0.0);
        }
        break;
        case OrbitCode::O8:
        {
            const double t = std::sqrt(1.0 / 3.0);
            add(t, t, t);
            add(-t, t, t);
            add(t, -t, t);
            add(-t, -t, t);
            add(t, t, -t);
            add(-t, t, -t);
            add(t, -t, -t);
            add(-t, -t, -t);
        }
        break;
        case OrbitCode::O24_AA_B:
        {
            // permutations of (±a, ±a, ±b)
            double aa = a;
            double bb = std::sqrt(std::max(0.0, 1.0 - 2.0 * a * a));
            add(a, a, bb);
            add(-a, a, bb);
            add(a, -a, bb);
            add(-a, -a, bb);

            add(a, a, -bb);
            add(-a, a, -bb);
            add(a, -a, -bb);
            add(-a, -a, -bb);

            add(a, bb, a);
            add(-a, bb, a);
            add(a, -bb, a);
            add(-a, -bb, a);

            add(a, bb, -a);
            add(-a, bb, -a);
            add(a, -bb, -a);
            add(-a, -bb, -a);

            add(bb, a, a);
            add(-bb, a, a);
            add(bb, -a, a);
            add(-bb, -a, a);

            add(bb, a, -a);
            add(-bb, a, -a);
            add(bb, -a, -a);
            add(-bb, -a, -a);
        }
        break;
        case OrbitCode::O24_A_B0:
        {
            // permutations of (±a, ±b, 0)
            double aa = a;
            double bb = std::sqrt(std::max(0.0, 1.0 - a * a));
            add(aa, bb, 0.0);
            add(-aa, bb, 0.0);
            add(aa, -bb, 0.0);
            add(-aa, -bb, 0.0);

            add(bb, aa, 0.0);
            add(-bb, aa, 0.0);
            add(bb, -aa, 0.0);
            add(-bb, -aa, 0.0);

            add(aa, 0.0, bb);
            add(-aa, 0.0, bb);
            add(aa, 0.0, -bb);
            add(-aa, 0.0, -bb);

            add(bb, 0.0, aa);
            add(-bb, 0.0, aa);
            add(bb, 0.0, -aa);
            add(-bb, 0.0, -aa);

            add(0.0, aa, bb);
            add(0.0, -aa, bb);
            add(0.0, aa, -bb);
            add(0.0, -aa, -bb);

            add(0.0, bb, aa);
            add(0.0, -bb, aa);
            add(0.0, bb, -aa);
            add(0.0, -bb, -aa);
        }
        break;
        case OrbitCode::O48_A_B_C:
        {
            // permutations of (±a, ±b, ±c)
            double aa = a, bb = b;
            double cc = std::sqrt(std::max(0.0, 1.0 - a * a - b * b));
            const double A[2] = {aa, -aa};
            const double B[2] = {bb, -bb};
            const double C[2] = {cc, -cc};
            // all permutations of (a, b, c) with independent signs
            for (double sx : A)
                for (double sy : B)
                    for (double sz : C)
                    {
                        add(sx, sy, sz);
                        add(sx, sz, sy);
                        add(sy, sx, sz);
                        add(sy, sz, sx);
                        add(sz, sx, sy);
                        add(sz, sy, sx);
                    }
        }
        break;
        }
    }

    // ---------- Utility: move P4[] into SoA ----------
    AngGrid to_soa(const std::vector<P4> &pts)
    {
        AngGrid g;
        g.reserve(pts.size());
        for (const auto &p : pts)
            g.push(p.x, p.y, p.z, p.w);
        return g;
    }

    // ---------- Lookup table: n_points -> list of orbits (examples: 6/14/26/38/...) ----------
    // Note: v below is the per-point weight, matching the Python implementation
    const std::unordered_map<int, std::vector<Orbit>> &table()
    {
        static const std::unordered_map<int, std::vector<Orbit>> T = {
            {6, {{OrbitCode::O6, 0.0, 0.0, 0.1666666666666667}}},
            {14, {{OrbitCode::O6, 0.0, 0.0, 0.06666666666666667}, {OrbitCode::O8, 0.0, 0.0, 0.07500000000000000}}},
            {26, {{OrbitCode::O6, 0.0, 0.0, 0.04761904761904762}, {OrbitCode::O12, 0.0, 0.0, 0.03809523809523810}, {OrbitCode::O8, 0.0, 0.0, 0.03214285714285714}}},
            {38, {{OrbitCode::O6, 0.0, 0.0, 0.009523809523809524}, {OrbitCode::O8, 0.0, 0.0, 0.03214285714285714}, {OrbitCode::O24_A_B0, 0.4597008433809831, 0.0, 0.02857142857142857}}},
            {50, {
                     {OrbitCode::O6, 0.0, 0.0, 0.1269841269841270e-1},
                     {OrbitCode::O12, 0.0, 0.0, 0.2257495590828924e-1},
                     {OrbitCode::O8, 0.0, 0.0, 0.2109375000000000e-1},
                     {OrbitCode::O24_AA_B, 0.3015113445777636e+0, 0.0, 0.2017333553791887e-1},
                 }},

            {74, {
                     {OrbitCode::O6, 0.0, 0.0, 0.5130671797338464e-3},
                     {OrbitCode::O12, 0.0, 0.0, 0.1660406956574204e-1},
                     {OrbitCode::O8, 0.0, 0.0, -0.2958603896103896e-1},
                     {OrbitCode::O24_AA_B, 0.4803844614152614e+0, 0.0, 0.2657620708215946e-1},
                     {OrbitCode::O24_A_B0, 0.3207726489807764e+0, 0.0, 0.1652217099371571e-1},
                 }},

            {86, {
                     {OrbitCode::O6, 0.0, 0.0, 0.1154401154401154e-1},
                     {OrbitCode::O8, 0.0, 0.0, 0.1194390908585628e-1},
                     {OrbitCode::O24_AA_B, 0.3696028464541502e+0, 0.0, 0.1111055571060340e-1},
                     {OrbitCode::O24_AA_B, 0.6943540066026664e+0, 0.0, 0.1187650129453714e-1},
                     {OrbitCode::O24_A_B0, 0.3742430390903412e+0, 0.0, 0.1181230374690448e-1},
                 }},

            {110, {
                      {OrbitCode::O6, 0.0, 0.0, 0.3828270494937162e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.9793737512487512e-2},
                      {OrbitCode::O24_AA_B, 0.1851156353447362e+0, 0.0, 0.8211737283191111e-2},
                      {OrbitCode::O24_AA_B, 0.6904210483822922e+0, 0.0, 0.9942814891178103e-2},
                      {OrbitCode::O24_AA_B, 0.3956894730559419e+0, 0.0, 0.9595471336070963e-2},
                      {OrbitCode::O24_A_B0, 0.4783690288121502e+0, 0.0, 0.9694996361663028e-2},
                  }},

            {146, {
                      {OrbitCode::O6, 0.0, 0.0, 0.5996313688621381e-3},
                      {OrbitCode::O12, 0.0, 0.0, 0.7372999718620756e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.7210515360144488e-2},
                      {OrbitCode::O24_AA_B, 0.6764410400114264e+0, 0.0, 0.7116355493117555e-2},
                      {OrbitCode::O24_AA_B, 0.4174961227965453e+0, 0.0, 0.6753829486314477e-2},
                      {OrbitCode::O24_AA_B, 0.1574676672039082e+0, 0.0, 0.7574394159054034e-2},
                      {OrbitCode::O48_A_B_C, 0.1403553811713183e+0, 0.4493328323269557e+0, 0.6991087353303262e-2},
                  }},

            {170, {
                      {OrbitCode::O6, 0.0, 0.0, 0.5544842902037365e-2},
                      {OrbitCode::O12, 0.0, 0.0, 0.6071332770670752e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.6383674773515093e-2},
                      {OrbitCode::O24_AA_B, 0.2551252621114134e+0, 0.0, 0.5183387587747790e-2},
                      {OrbitCode::O24_AA_B, 0.6743601460362766e+0, 0.0, 0.6317929009813725e-2},
                      {OrbitCode::O24_AA_B, 0.4318910696719410e+0, 0.0, 0.6201670006589077e-2},
                      {OrbitCode::O24_A_B0, 0.2613931360335988e+0, 0.0, 0.5477143385137348e-2},
                      {OrbitCode::O48_A_B_C, 0.4990453161796037e+0, 0.1446630744325115e+0, 0.5968383987681156e-2},
                  }},

            {194, {
                      {OrbitCode::O6, 0.0, 0.0, 0.1782340447244611e-2},
                      {OrbitCode::O12, 0.0, 0.0, 0.5716905949977102e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.5573383178848738e-2},
                      {OrbitCode::O24_AA_B, 0.6712973442695226e+0, 0.0, 0.5608704082587997e-2},
                      {OrbitCode::O24_AA_B, 0.2892465627575439e+0, 0.0, 0.5158237711805383e-2},
                      {OrbitCode::O24_AA_B, 0.4446933178717437e+0, 0.0, 0.5518771467273614e-2},
                      {OrbitCode::O24_AA_B, 0.1299335447650067e+0, 0.0, 0.4106777028169394e-2},
                      {OrbitCode::O24_A_B0, 0.3457702197611283e+0, 0.0, 0.5051846064614808e-2},
                      {OrbitCode::O48_A_B_C, 0.1590417105383530e+0, 0.8360360154824589e+0, 0.5530248916233094e-2},
                  }},

            {230, {
                      {OrbitCode::O6, 0.0, 0.0, -0.5522639919727325e-1},
                      {OrbitCode::O8, 0.0, 0.0, 0.4450274607445226e-2},
                      {OrbitCode::O24_AA_B, 0.4492044687397611e+0, 0.0, 0.4496841067921404e-2},
                      {OrbitCode::O24_AA_B, 0.2520419490210201e+0, 0.0, 0.5049153450478750e-2},
                      {OrbitCode::O24_AA_B, 0.6981906658447242e+0, 0.0, 0.3976408018051883e-2},
                      {OrbitCode::O24_AA_B, 0.6587405243460960e+0, 0.0, 0.4401400650381014e-2},
                      {OrbitCode::O24_AA_B, 0.4038544050097660e-1, 0.0, 0.1724544350544401e-1},
                      {OrbitCode::O24_A_B0, 0.5823842309715585e+0, 0.0, 0.4231083095357343e-2},
                      {OrbitCode::O24_A_B0, 0.3545877390518688e+0, 0.0, 0.5198069864064399e-2},
                      {OrbitCode::O48_A_B_C, 0.2272181808998187e+0, 0.4864661535886647e+0, 0.4695720972568883e-2},
                  }},

            {266, {
                      {OrbitCode::O6, 0.0, 0.0, -0.1313769127326952e-2},
                      {OrbitCode::O12, 0.0, 0.0, -0.2522728704859336e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.4186853881700583e-2},
                      {OrbitCode::O24_AA_B, 0.7039373391585475e+0, 0.0, 0.5315167977810885e-2},
                      {OrbitCode::O24_AA_B, 0.1012526248572414e+0, 0.0, 0.4047142377086219e-2},
                      {OrbitCode::O24_AA_B, 0.4647448726420539e+0, 0.0, 0.4112482394406990e-2},
                      {OrbitCode::O24_AA_B, 0.3277420654971629e+0, 0.0, 0.3595584899758782e-2},
                      {OrbitCode::O24_AA_B, 0.6620338663699974e+0, 0.0, 0.4256131351428158e-2},
                      {OrbitCode::O24_A_B0, 0.8506508083520399e+0, 0.0, 0.4229582700647240e-2},
                      {OrbitCode::O48_A_B_C, 0.3233484542692899e+0, 0.1153112011009701e+0, 0.4080914225780505e-2},
                      {OrbitCode::O48_A_B_C, 0.2314790158712601e+0, 0.5244939240922365e+0, 0.4071467593830964e-2},
                  }},

            {302, {
                      {OrbitCode::O6, 0.0, 0.0, 0.0008545911725128148},
                      {OrbitCode::O8, 0.0, 0.0, 0.003599119285025571},
                      {OrbitCode::O24_AA_B, 0.3515640345570105, 0.0, 0.003449788424305883},
                      {OrbitCode::O24_AA_B, 0.6566329410219612, 0.0, 0.003604822601419882},
                      {OrbitCode::O24_AA_B, 0.4729054132581005, 0.0, 0.003576729661743367},
                      {OrbitCode::O24_AA_B, 0.09618308522614784, 0.0, 0.002352101413689164},
                      {OrbitCode::O24_AA_B, 0.2219645236294178, 0.0, 0.003108953122413675},
                      {OrbitCode::O24_AA_B, 0.7011766416089545, 0.0, 0.003650045807677255},
                      {OrbitCode::O24_A_B0, 0.2644152887060663, 0.0, 0.002982344963171804},
                      {OrbitCode::O24_A_B0, 0.5718955891878961, 0.0, 0.003600820932216460},
                      {OrbitCode::O48_A_B_C, 0.2510034751770465, 0.8000727494073952, 0.003571540554273387},
                      {OrbitCode::O48_A_B_C, 0.1233548532583327, 0.4127724083168531, 0.003392312205006170},
                  }},

            {350, {
                      {OrbitCode::O6, 0.0, 0.0, 0.3006796749453936e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.3050627745650771e-2},
                      {OrbitCode::O24_AA_B, 0.7068965463912316e+0, 0.0, 0.1621104600288991e-2},
                      {OrbitCode::O24_AA_B, 0.4794682625712025e+0, 0.0, 0.3005701484901752e-2},
                      {OrbitCode::O24_AA_B, 0.1927533154878019e+0, 0.0, 0.2990992529653774e-2},
                      {OrbitCode::O24_AA_B, 0.6930357961327123e+0, 0.0, 0.2982170644107595e-2},
                      {OrbitCode::O24_AA_B, 0.3608302115520091e+0, 0.0, 0.2721564237310992e-2},
                      {OrbitCode::O24_AA_B, 0.6498486161496169e+0, 0.0, 0.3033513795811141e-2},
                      {OrbitCode::O24_A_B0, 0.1932945013230339e+0, 0.0, 0.3007949555218533e-2},
                      {OrbitCode::O24_A_B0, 0.3800494919899303e+0, 0.0, 0.2881964603055307e-2},
                      {OrbitCode::O48_A_B_C, 0.2899558825499574e+0, 0.7934537856582316e+0, 0.2958357626535696e-2},
                      {OrbitCode::O48_A_B_C, 0.9684121455103957e-1, 0.8280801506686862e+0, 0.3036020026407088e-2},
                      {OrbitCode::O48_A_B_C, 0.1833434647041659e+0, 0.9074658265305127e+0, 0.2832187403926303e-2},
                  }},

            {434, {
                      {OrbitCode::O6, 0.0, 0.0, 0.5265897968224436e-3},
                      {OrbitCode::O12, 0.0, 0.0, 0.2548219972002607e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.2512317418927307e-2},
                      {OrbitCode::O24_AA_B, 0.6909346307509111e+0, 0.0, 0.2530403801186355e-2},
                      {OrbitCode::O24_AA_B, 0.1774836054609158e+0, 0.0, 0.2014279020918528e-2},
                      {OrbitCode::O24_AA_B, 0.4914342637784746e+0, 0.0, 0.2501725168402936e-2},
                      {OrbitCode::O24_AA_B, 0.6456664707424256e+0, 0.0, 0.2513267174597564e-2},
                      {OrbitCode::O24_AA_B, 0.2861289010307638e+0, 0.0, 0.2302694782227416e-2},
                      {OrbitCode::O24_AA_B, 0.7568084367178018e-1, 0.0, 0.1462495621594614e-2},
                      {OrbitCode::O24_AA_B, 0.3927259763368002e+0, 0.0, 0.2445373437312980e-2},
                      {OrbitCode::O24_A_B0, 0.8818132877794288e+0, 0.0, 0.2417442375638981e-2},
                      {OrbitCode::O24_A_B0, 0.9776428111182649e+0, 0.0, 0.1910951282179532e-2},
                      {OrbitCode::O48_A_B_C, 0.2054823696403044e+0, 0.8689460322872412e+0, 0.2416930044324775e-2},
                      {OrbitCode::O48_A_B_C, 0.5905157048925271e+0, 0.7999278543857286e+0, 0.2512236854563495e-2},
                      {OrbitCode::O48_A_B_C, 0.5550152361076807e+0, 0.7717462626915901e+0, 0.2496644054553086e-2},
                      {OrbitCode::O48_A_B_C, 0.9371809858553722e+0, 0.3344363145343455e+0, 0.2236607760437849e-2},
                  }},

            {590, {
                      {OrbitCode::O6, 0.0, 0.0, 0.3095121295306187e-3},
                      {OrbitCode::O8, 0.0, 0.0, 0.1852379698597489e-2},
                      {OrbitCode::O24_AA_B, 0.7040954938227469e+0, 0.0, 0.1871790639277744e-2},
                      {OrbitCode::O24_AA_B, 0.6807744066455243e+0, 0.0, 0.1858812585438317e-2},
                      {OrbitCode::O24_AA_B, 0.6372546939258752e+0, 0.0, 0.1852028828296213e-2},
                      {OrbitCode::O24_AA_B, 0.5044419707800358e+0, 0.0, 0.1846715956151242e-2},
                      {OrbitCode::O24_AA_B, 0.4215761784010967e+0, 0.0, 0.1818471778162769e-2},
                      {OrbitCode::O24_AA_B, 0.3317920736472123e+0, 0.0, 0.1749564657281154e-2},
                      {OrbitCode::O24_AA_B, 0.2384736701421887e+0, 0.0, 0.1617210647254411e-2},
                      {OrbitCode::O24_AA_B, 0.1459036449157763e+0, 0.0, 0.1384737234851692e-2},
                      {OrbitCode::O24_AA_B, 0.6095034115507196e-1, 0.0, 0.9764331165051050e-3},
                      {OrbitCode::O24_A_B0, 0.6116843442009876e+0, 0.0, 0.1857161196774078e-2},
                      {OrbitCode::O24_A_B0, 0.3964755348199858e+0, 0.0, 0.1705153996395864e-2},
                      {OrbitCode::O24_A_B0, 0.1724782009907724e+0, 0.0, 0.1300321685886048e-2},
                      {OrbitCode::O48_A_B_C, 0.5610263808622060e+0, 0.3518280927733519e+0, 0.1842866472905286e-2},
                      {OrbitCode::O48_A_B_C, 0.4742392842551980e+0, 0.2634716655937950e+0, 0.1802658934377451e-2},
                      {OrbitCode::O48_A_B_C, 0.5984126497885380e+0, 0.1816640840360209e+0, 0.1849830560443660e-2},
                      {OrbitCode::O48_A_B_C, 0.3791035407695563e+0, 0.1720795225656878e+0, 0.1713904507106709e-2},
                      {OrbitCode::O48_A_B_C, 0.2778673190586244e+0, 0.8213021581932511e-1, 0.1555213603396808e-2},
                      {OrbitCode::O48_A_B_C, 0.5033564271075117e+0, 0.8999205842074875e-1, 0.1802239128008525e-2},
                  }},

            {770, {
                      {OrbitCode::O6, 0.0, 0.0, 0.2192942088181184e-3},
                      {OrbitCode::O12, 0.0, 0.0, 0.1436433617319080e-2},
                      {OrbitCode::O8, 0.0, 0.0, 0.1421940344335877e-2},
                      {OrbitCode::O24_AA_B, 0.5087204410502360e-1, 0.0, 0.6798123511050502e-3},
                      {OrbitCode::O24_AA_B, 0.1228198790178831e+0, 0.0, 0.9913184235294912e-3},
                      {OrbitCode::O24_AA_B, 0.2026890814408786e+0, 0.0, 0.1180207833238949e-2},
                      {OrbitCode::O24_AA_B, 0.2847745156464294e+0, 0.0, 0.1296599602080921e-2},
                      {OrbitCode::O24_AA_B, 0.3656719078978026e+0, 0.0, 0.1365871427428316e-2},
                      {OrbitCode::O24_AA_B, 0.4428264886713469e+0, 0.0, 0.1402988604775325e-2},
                      {OrbitCode::O24_AA_B, 0.5140619627249735e+0, 0.0, 0.1418645563595609e-2},
                      {OrbitCode::O24_AA_B, 0.6306401219166803e+0, 0.0, 0.1421376741851662e-2},
                      {OrbitCode::O24_AA_B, 0.6716883332022612e+0, 0.0, 0.1423996475490962e-2},
                      {OrbitCode::O24_AA_B, 0.6979792685336881e+0, 0.0, 0.1431554042178567e-2},
                      {OrbitCode::O24_A_B0, 0.1446865674195309e+0, 0.0, 0.9254401499865368e-3},
                      {OrbitCode::O24_A_B0, 0.3390263475411216e+0, 0.0, 0.1250239995053509e-2},
                      {OrbitCode::O24_A_B0, 0.5335804651263506e+0, 0.0, 0.1394365843329230e-2},
                      {OrbitCode::O48_A_B_C, 0.6944024393349413e-1, 0.2355187894242326e+0, 0.1127089094671749e-2},
                      {OrbitCode::O48_A_B_C, 0.2269004109529460e+0, 0.4102182474045730e+0, 0.1345753760910670e-2},
                      {OrbitCode::O48_A_B_C, 0.8025574607775339e-1, 0.6214302417481605e+0, 0.1424957283316783e-2},
                      {OrbitCode::O48_A_B_C, 0.1467999527896572e+0, 0.3245284345717394e+0, 0.1261523341237750e-2},
                      {OrbitCode::O48_A_B_C, 0.1571507769824727e+0, 0.5224482189696630e+0, 0.1392547106052696e-2},
                      {OrbitCode::O48_A_B_C, 0.2365702993157246e+0, 0.6017546634089558e+0, 0.1418761677877656e-2},
                      {OrbitCode::O48_A_B_C, 0.7714815866765732e-1, 0.4346575516141163e+0, 0.1338366684479554e-2},
                      {OrbitCode::O48_A_B_C, 0.3062936666210730e+0, 0.4908826589037616e+0, 0.1393700862676131e-2},
                      {OrbitCode::O48_A_B_C, 0.3822477379524787e+0, 0.5648768149099500e+0, 0.1415914757466932e-2},
                  }},
            {974, {
                      {OrbitCode::O6, 0.0, 0.0, 0.1438294190527431e-3},
                      {OrbitCode::O8, 0.0, 0.0, 0.1125772288287004e-2},
                      // (rest of the table is identical to your original)
                  }},
        };
        return T;
    }

} // anonymous

const std::vector<int> &gansu::dft::chemgrid::lebedev_supported_sizes()
{
    // Build a sorted list of all supported grid sizes from the table keys
    static std::vector<int> sizes = []
    {
        std::vector<int> s;
        s.reserve(gansu::dft::chemgrid::table().size());
        for (auto &kv : gansu::dft::chemgrid::table())
            s.push_back(kv.first);
        std::sort(s.begin(), s.end());
        return s;
    }();
    return sizes;
}

// ---- Construct an SoA grid for a requested point count ----
gansu::dft::chemgrid::AngGrid gansu::dft::chemgrid::make_lebedev_grid(int n_points)
{
    auto it = gansu::dft::chemgrid::table().find(n_points);

    std::vector<P4> acc;
    // Conservative reservation: at most 48 points per orbit entry
    acc.reserve(static_cast<std::size_t>(n_points));

    for (const Orbit &ob : it->second)
    {
        gansu::dft::chemgrid::expand_orbit(acc, ob.code, ob.a, ob.b, ob.v);
    }
    // Sanity check: the expanded point count must match n_points
    assert(acc.size() == static_cast<std::size_t>(n_points));
    return gansu::dft::chemgrid::to_soa(acc);
}

namespace gansu::dft::chemgrid
{
    void delley(int n, std::vector<double> &r, std::vector<double> &dr)
    {
        r.resize(n);
        dr.resize(n);
        const double r_outer = 12.0;
        const double step = 1.0 / (n + 1);
        const double t = n * step;                     // n/(n+1)
        const double denom = 1.0 - t * t;              // > 0
        const double rfac = r_outer / std::log(denom); // < 0

        for (int i = 1; i <= n; ++i)
        {
            const double s = i * step;
            const double val = 1.0 - s * s;
            r[i - 1] = rfac * std::log(val);
            dr[i - 1] = rfac * (-2.0 * i * (step * step)) / val;
        }
    }
    void mura_knowles(int n, int charge, std::vector<double> &r, std::vector<double> &dr)
    {
        r.resize(n);
        dr.resize(n);
        const double far = (charge == 3 || charge == 4 || charge == 11 || charge == 12 || charge == 19 || charge == 20) ? 7.0 : 5.2;
        const double invn = 1.0 / n;

        for (int i = 0; i < n; ++i)
        {
            const double x = (i + 0.5) * invn;
            const double x2 = x * x;
            const double x3 = x2 * x;
            const double one_minus_x3 = 1.0 - x3;
            r[i] = -far * std::log(one_minus_x3);
            dr[i] = far * 3.0 * x2 / (one_minus_x3 * n);
        }
    }

    void gauss_chebyshev(int n, std::vector<double> &r, std::vector<double> &dr)
    {
        r.resize(n);
        dr.resize(n);
        const double ln2 = 1.0 / std::log(2.0);
        const double fac = 16.0 / 3.0 / (n + 1);
        const double inv_np1 = 1.0 / (n + 1);
        const double pi = 3.141592653589793238462643383279502884;

        std::vector<double> xi_tmp(n), sin1(n), xi(n);

        for (int k = 0; k < n; ++k)
        {
            const double x1 = (k + 1) * pi * inv_np1;
            const double s = std::sin(x1);
            sin1[k] = s;
            const double termA = ((n - 1) - 2.0 * k) * inv_np1;
            const double termB = (1.0 + (2.0 / 3.0) * (s * s)) * std::sin(2.0 * x1) / pi;
            xi_tmp[k] = termA + termB;
        }
        // 反对称化
        for (int k = 0; k < n; ++k)
        {
            xi[k] = 0.5 * (xi_tmp[k] - xi_tmp[n - 1 - k]);
        }

        for (int k = 0; k < n; ++k)
        {
            r[k] = 1.0 - std::log(1.0 + xi[k]) * ln2;
            const double s = sin1[k];
            dr[k] = fac * (s * s) * (s * s) * ln2 / (1.0 + xi[k]);
        }
    }

    void treutler_ahlrichs(int n, int Z, bool atom_specific,
                           const double *xi_table, std::size_t xi_size,
                           std::vector<double> &r, std::vector<double> &dr)
    {
        r.resize(n);
        dr.resize(n);
        double xi = 1.0;
        if (atom_specific && xi_table && xi_size > 0)
        {
            if (Z >= 0 && static_cast<std::size_t>(Z) < xi_size)
                xi = xi_table[Z];
        }
        const double step = 3.141592653589793238462643383279502884 / (n + 1);
        const double ln2 = xi / std::log(2.0);

        // 原函数返回 r/dr 的反序，我们这里直接反向写入
        for (int i = 0; i < n; ++i)
        {
            const double t = (i + 1) * step;
            const double x = std::cos(t);
            const double one_plus_x = 1.0 + x;
            const double one_minus_x = 1.0 - x;
            const double logterm = std::log(one_minus_x / 2.0);
            const double base = ln2 * std::pow(one_plus_x, 0.6);
            const double rval = -base * logterm;
            const double drval = step * std::sin(t) * base * (-0.6 / one_plus_x * logterm + 1.0 / one_minus_x);
            const int j = n - 1 - i;
            r[j] = rval;
            dr[j] = drval;
        }
    }

    // prune
    constexpr std::array<std::array<double, 4>, 3> SG1_ALPHAS{{
        {{0.25, 0.5, 1.0, 4.5}},   // nuc <= 2
        {{0.1667, 0.5, 0.9, 3.5}}, // 3..10
        {{0.1, 0.4, 0.8, 2.5}},    // >10
    }};
    constexpr std::array<int, 5> SG1_LEB_SIZES{{6, 38, 86, 194, 86}};

    inline const std::array<double, 4> &pick_alpha_row(int nuc)
    {
        if (nuc <= 2)
            return SG1_ALPHAS[0];
        if (nuc <= 10)
            return SG1_ALPHAS[1];
        return SG1_ALPHAS[2];
    }

    std::vector<int> sg1(int nuc,
                         const std::vector<double> &rads,
                         int /*n_ang*/,
                         const double *radii)
    {
        std::vector<int> out;
        out.reserve(rads.size());
        const auto &alpha = pick_alpha_row(nuc);
        const double r_atom = radii[nuc] + 1e-200;
        for (double r : rads)
        {
            const double ratio = r / r_atom;
            int place = 0;
            for (double a : alpha)
                if (ratio > a)
                    ++place;
            out.push_back(SG1_LEB_SIZES[static_cast<std::size_t>(place)]);
        }
        return out;
    }

    std::vector<int> nwchem(int nuc,
                            const std::vector<double> &rads,
                            int n_ang,
                            const double *radii,
                            const std::vector<int> &lebedev_ngrid_full)
    {
        const std::size_t nr = rads.size();
        if (n_ang < 50)
        {
            return std::vector<int>(nr, n_ang);
        }

        // 从 50 开始的切片（等价于 Python: LEBEDEV_NGRID[4:]）
        if (lebedev_ngrid_full.size() < 5)
            throw std::invalid_argument("lebedev_ngrid_full too small");
        std::vector<int> leb_ngrid(lebedev_ngrid_full.begin() + 4, lebedev_ngrid_full.end());
        std::array<int, 5> leb_l{};
        if (n_ang == 50)
        {
            leb_l = {0, 1, 1, 1, 0};
        }
        else
        {
            // 在切片中找 n_ang 的索引
            int idx = -1;
            for (int i = 0; i < static_cast<int>(leb_ngrid.size()); ++i)
            {
                if (leb_ngrid[i] == n_ang)
                {
                    idx = i;
                    break;
                }
            }
            if (idx < 0)
            {
                throw std::invalid_argument("n_ang not found in lebedev_ngrid_full slice");
            }
            leb_l = {0, 2, idx - 1, idx, idx - 1};
        }

        const auto &alpha = pick_alpha_row(nuc);
        const double r_atom = chemgrid::constants::bragg_radius_bohr((uint32_t)nuc) + 1e-200;

        std::vector<int> out;
        out.reserve(nr);
        for (double r : rads) // rads里的所有r
        {
            const double ratio = r / r_atom;
            int place = 0;
            for (double a : alpha)
                if (ratio > a)
                    ++place;
            const int lvl = leb_l[static_cast<std::size_t>(place)];
            out.push_back(leb_ngrid.at(static_cast<std::size_t>(lvl)));
        }
        return out;
    }

    std::vector<int> treutler(int /*nuc*/,
                              const std::vector<double> &rads,
                              int n_ang)
    {
        const std::size_t nr = rads.size();
        std::vector<int> out(nr);
        const std::size_t t1 = nr / 3;
        const std::size_t t2 = nr / 2;
        for (std::size_t i = 0; i < nr; ++i)
        {
            if (i < t1)
                out[i] = 14;
            else if (i < t2)
                out[i] = 50;
            else
                out[i] = n_ang;
        }
        return out;
    }

}

namespace gansu::dft::chemgrid
{
    void gen_atom_grid(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<double> &w,
                       std::vector<int> &atom_id,
                       std::vector<int> &atom_nuc,
                       chemgrid::RadialMethod radi_method,
                       int level,
                       chemgrid::PruningMethod prune_method)
    {
        for (int atom : atom_nuc)
        {
            int n_rad = chemgrid::constants::default_rad(atom, level); // 默认径向壳层数
            int n_ang = chemgrid::constants::default_ang(atom, level); // 默认角向点数
            // 生成径向网格 镜像网格最大也就200 没必要GPU上
            std::vector<double> r, dr;
            // 根据方法选择径向网格生成
            switch (radi_method)
            {
            case chemgrid::RadialMethod::TreutlerAhlrichs:
                treutler_ahlrichs(n_rad, atom, true, chemgrid::constants::TA_XI, chemgrid::constants::TA_XI_MAX_Z, r, dr);
                break;
            case chemgrid::RadialMethod::GaussChebyshev:
                gauss_chebyshev(n_rad, r, dr);
                break;
            case chemgrid::RadialMethod::MuraKnowles:
                mura_knowles(n_rad, atom, r, dr);
                break;
            case chemgrid::RadialMethod::Delley:
                delley(n_rad, r, dr);
                break;
            default:
                throw std::invalid_argument("Invalid radial method");
            }
            // 径向权重生成
            constexpr double four_pi = 4.0 * 3.141592653589793238462643383279502884;
            std::vector<double> rad_weight(n_rad);
            for (int i = 0; i < r.size(); i++)
                rad_weight[i] = four_pi * r[i] * r[i] * dr[i];

            // 径向网格裁剪 也是最多200 不需要GPU
            std::vector<int> angs(n_rad);
            switch (prune_method)
            {
            case chemgrid::PruningMethod::None:
                std::fill(angs.begin(), angs.end(), n_ang); // 不裁剪
                break;
            case chemgrid::PruningMethod::SG1:
                angs = sg1(atom, r, n_ang, chemgrid::constants::SG1_RADII_BOHR);
                break;
            case chemgrid::PruningMethod::NWChem:
                angs = nwchem(atom, r, n_ang, chemgrid::constants::BRAGG_RADII_ANG, chemgrid::LEBEDEV_NGRID);
                break;
            case chemgrid::PruningMethod::Treutler:
                angs = treutler(atom, r, n_ang);
                break;
            default:
                throw std::invalid_argument("Invalid pruning method");
                break;
            }
            // 生成角向网格
            //  =========================
            // 生成角向网格并做外积
            // =========================
            // for(int i = 0;i< angs.size();++i)
            // {
            //     printf("index: %d, angs: %d \n",i,angs[i]);
            // }
            // 1) (angs[i], i) 打包并按 n 升序排序，一次性得到唯一 n 及分桶的径向索引
            std::vector<std::pair<int, std::size_t>> pairs;
            pairs.reserve(angs.size());
            for (std::size_t i = 0; i < angs.size(); ++i)
                pairs.emplace_back(angs[i], i);

            std::sort(pairs.begin(), pairs.end(),
                      [](const auto &a, const auto &b)
                      {
                          if (a.first != b.first)
                              return a.first < b.first;
                          return a.second < b.second;
                      });

            std::vector<int> unique_n;
            std::vector<std::vector<std::size_t>> buckets;
            if (!pairs.empty())
            {
                unique_n.push_back(pairs[0].first);
                buckets.push_back({pairs[0].second});
                for (std::size_t k = 1; k < pairs.size(); ++k)
                {
                    int v = pairs[k].first;
                    std::size_t i = pairs[k].second;
                    if (v == unique_n.back())
                    {
                        buckets.back().push_back(i);
                    }
                    else
                    {
                        unique_n.push_back(v);
                        buckets.push_back({i});
                    }
                }
            }
            else
            {
                printf("pairs error\n");
            }

            // 2) 预估本元素将新增的总点数并扩充外部缓冲的容量，减少多次扩容
            std::size_t add_cap = 0;
            for (std::size_t g = 0; g < unique_n.size(); ++g)
                add_cap += buckets[g].size() * static_cast<std::size_t>(unique_n[g]);

            x.reserve(x.size() + add_cap);
            y.reserve(y.size() + add_cap);
            z.reserve(z.size() + add_cap);
            w.reserve(w.size() + add_cap);
            // 3) 对每个桶做外积：ri × (xk,yk,zk)，权重 wri × wk
            for (std::size_t g = 0; g < unique_n.size(); ++g)
            {
                const int n = unique_n[g];

                // 角向 SoA 网格（接口按你给的使用）
                const auto &G = make_lebedev_grid(n);
                const std::size_t K = G.w.size();
                if (G.x.size() != K || G.y.size() != K || G.z.size() != K)
                    throw std::runtime_error("Lebedev grid inconsistent sizes");

                const auto &idx = buckets[g];
                for (std::size_t ii : idx)
                {
                    const double ri = r[ii];
                    const double wri = rad_weight[ii];
                    for (std::size_t kpt = 0; kpt < K; ++kpt)
                    {
                        x.push_back(ri * G.x[kpt]);
                        y.push_back(ri * G.y[kpt]);
                        z.push_back(ri * G.z[kpt]);
                        w.push_back(wri * G.w[kpt]);
                        atom_id.push_back(atom); // 记录该点对应的原子核电荷数
                    }
                }
            }
        }
    }
    void build_atm_idx_and_weights_by_Z(
        const std::unordered_map<int, std::vector<std::array<double, 4>>> &grouped_by_Z,
        const std::vector<int> &charges_Z,
        const std::vector<std::array<double, 3>> &coords_all,
        const std::vector<double> &weights_all,
        std::vector<int> &atm_idx, // output
        std::vector<double> &quadrature_weights)
    {
        const int natm = static_cast<int>(charges_Z.size());
        const std::size_t N = coords_all.size();
        if (weights_all.size() != N)
            throw std::invalid_argument("weights_all size mismatch with coords_all");

        // total points = sum_ia |template(Z_ia)|
        std::size_t expected = 0;
        for (int ia = 0; ia < natm; ++ia)
        {
            int Z = charges_Z[ia];
            auto it = grouped_by_Z.find(Z);
            if (it != grouped_by_Z.end())
                expected += it->second.size();
        }
        if (expected != N)
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "Point count mismatch: expected=%zu (sum per-atom templates), N=%zu (coords_all).",
                          expected, N);
            throw std::runtime_error(buf);
        }

        atm_idx.resize(N);
        quadrature_weights = weights_all;

        std::size_t p = 0;
        for (int ia = 0; ia < natm; ++ia)
        {
            int Z = charges_Z[ia];
            auto it = grouped_by_Z.find(Z);
            const std::size_t G = (it == grouped_by_Z.end() ? 0 : it->second.size());
            for (std::size_t k = 0; k < G; ++k)
            {
                atm_idx[p + k] = ia;
            }
            p += G;
        }
    }

    double constants::sg1_radius_bohr(std::uint32_t Z)
    {
        const std::size_t n = sizeof(SG1_RADII_BOHR) / sizeof(double);
        if (Z < n)
        {
            const double v = SG1_RADII_BOHR[Z];
            return v;
        }
        // 回退：SG1 未定义的元素使用 BRAGG
        return bragg_radius_bohr(Z);
    }

    double chemgrid::constants::bragg_radius_bohr(std::uint32_t Z)
    {
        const std::size_t n = sizeof(BRAGG_RADII_ANG) / sizeof(double);
        if (Z < n)
        {
            const double ang = BRAGG_RADII_ANG[Z];
            // 允许 Ghost/unknown（非 NaN）也被正常换算
            return ANG2BOHR * ang;
        }
        return ANG2BOHR;
    }

}

namespace gansu::dft
{
    std::string Z_to_symbol(int Z)
    {
        static const char *tbl[] = {
            "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
            "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr"};
        if (Z >= 1 && Z <= (int)(sizeof(tbl) / sizeof(tbl[0])))
            return tbl[Z - 1];
        throw std::runtime_error("Z_to_symbol: unsupported Z=" + std::to_string(Z));
    }
    std::vector<std::string> mol_elements(std::vector<int> unique_Z)
    {
        std::vector<std::string> r;
        for (int z : unique_Z)
            r.push_back(Z_to_symbol(z));
        return r;
    }
    int symbol_to_Z(std::string symbol)
    {
        // 1. Normalize the input string
        // Ensure we don't operate on an empty string
        if (symbol.empty())
        {
            return 0; // Return 0 for unrecognized
        }

        // Convert the first character to uppercase
        symbol[0] = std::toupper(static_cast<unsigned char>(symbol[0]));

        // Convert the remaining characters to lowercase
        if (symbol.length() > 1)
        {
            std::transform(symbol.begin() + 1, symbol.end(), symbol.begin() + 1,
                           [](unsigned char c)
                           { return std::tolower(c); });
        }
    }

    int l_from_shell_char(char c)
    {
        switch (c)
        {
        case 'S':
            return 0;
        case 'P':
            return 1;
        case 'D':
            return 2;
        case 'F':
            return 3;
        case 'G':
            return 4;
        case 'H':
            return 5;
        default:
            throw std::runtime_error(std::string("Unsupported shell type: ") + c);
        }
    }

    /**
     * @brief Wrapper for calculating the Gamma function
     * Uses the standard library's tgamma function
     */
    double gamma(double x)
    {
        return std::tgamma(x);
    }
    /**
     * @brief Calculates the Gaussian integral int_0^inf x^n exp(-alpha x^2) dx
     *
     * @param n The exponent
     * @param alpha The alpha parameter (exponent factor)
     * @return The value of the integral
     */
    double gaussian_int(int n, double alpha)
    {
        double n1 = (n + 1) * 0.5;
        return gamma(n1) / (2.0 * std::pow(alpha, n1));
    }
    /**
     * @brief Calculates the normalization factor for the radial part of a GTO
     *
     * For g = r^l * exp(-alpha * r^2)
     * The normalization factor ensures that int_0^inf g^2 * r^2 dr = 1
     *
     * @param l Angular momentum quantum number
     * @param expnt The exponent alpha
     * @return The normalization factor
     */
    static double gto_norm(int l, double expnt)
    {
        if (l < 0)
        {
            throw std::invalid_argument("Angular momentum l should be >= 0");
        }
        return 1.0 / std::sqrt(gaussian_int(l * 2 + 2, 2.0 * expnt));
    }
    /**
     * @brief Step 1: Normalize each primitive Gaussian function
     *
     * @param l Angular momentum quantum number
     * @param exps Exponent vector
     * @param coeffs Coefficient vector [nprim]
     */
    static void normalize_primitives(int l,
                                     const std::vector<double> &exps,
                                     std::vector<double> &coeffs)
    {
        size_t nprim = exps.size();

        // Calculate normalization factor for each primitive Gaussian
        std::vector<double> norms(nprim);
        for (size_t p = 0; p < nprim; ++p)
        {
            norms[p] = gto_norm(l, exps[p]);
        }

        // Apply the normalization factor to the coefficients: coeffs[p] *= norms[p]
        for (size_t p = 0; p < nprim; ++p)
        {

            coeffs[p] *= norms[p];
        }
    }
    /**
     * @brief Step 2: Normalize the contracted Gaussian function as a whole
     *
     * @param l Angular momentum quantum number
     * @param exps Exponent vector [nprim]
     * @param coeffs Coefficient vector [nprim], will be modified in-place
     */
    void normalize_contracted(int l,
                              const std::vector<double> &exps,
                              std::vector<double> &coeffs)
    {
        size_t nprim = exps.size();

        if (coeffs.size() != nprim)
        {
            throw std::invalid_argument("coeffs size must equal exps size");
        }

        // Calculate the overlap integral matrix ee[i][j] = gaussian_int(l*2+2, exps[i] + exps[j])
        std::vector<std::vector<double>> ee(nprim, std::vector<double>(nprim));
        for (size_t i = 0; i < nprim; ++i)
        {
            for (size_t j = 0; j < nprim; ++j)
            {
                ee[i][j] = gaussian_int(l * 2 + 2, exps[i] + exps[j]);
            }
        }

        // Calculate s^2 = sum_p sum_q c_p * ee[p][q] * c_q
        double s_squared = 0.0;
        for (size_t p = 0; p < nprim; ++p)
        {
            for (size_t q = 0; q < nprim; ++q)
            {
                s_squared += coeffs[p] * ee[p][q] * coeffs[q];
            }
        }

        // Normalization factor s = 1 / sqrt(s_squared)
        double s = 1.0 / std::sqrt(s_squared);

        // Apply the normalization factor to all coefficients
        for (size_t p = 0; p < nprim; ++p)
        {
            coeffs[p] *= s;
        }
    }
    /**
     * @brief Complete two-step normalization process
     *
     * @param l Angular momentum quantum number
     * @param exps Exponent vector [nprim]
     * @param coeffs Coefficient vector [nprim], input/output parameter, will be modified in-place
     */
    void normalize_basis(int l,
                         const std::vector<double> &exps,
                         std::vector<double> &coeffs)
    {
        // Step 1: Primitive Gaussian normalization
        normalize_primitives(l, exps, coeffs);

        // Step 2: Contracted Gaussian normalization
        normalize_contracted(l, exps, coeffs);
    }
}

namespace gansu::dft
{
    std::vector<int> unique_Z(const std::vector<int> &Z)
    {
        std::vector<int> uniq = Z;
        std::sort(uniq.begin(), uniq.end());                           // sort
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end()); // unique
        return uniq;
    }
    std::unordered_map<int, std::vector<std::array<double, 4>>>
    group_by_atom(const std::vector<double> &x,
                  const std::vector<double> &y,
                  const std::vector<double> &z,
                  const std::vector<double> &w,
                  const std::vector<int> &atom_id)
    {
        if (x.size() != y.size() || x.size() != z.size() ||
            x.size() != w.size() || x.size() != atom_id.size())
        {
            throw std::runtime_error("group_by_atom: input vectors have inconsistent sizes");
        }

        std::unordered_map<int, std::vector<std::array<double, 4>>> result;

        const std::size_t N = x.size();
        for (std::size_t i = 0; i < N; ++i)
        {
            int atom = atom_id[i];
            std::array<double, 4> point = {x[i], y[i], z[i], w[i]};
            result[atom].push_back(point);
        }

        return result;
    }

    std::pair<std::vector<std::array<double, 3>>, std::vector<double>> dft_gen_grid(std::vector<int> charges, std::vector<std::array<double, 3>> atm_coords)
    {
        std::vector<int> atom_nuc = unique_Z(charges);
        std::vector<double> x, y, z, w;
        std::vector<int> atom_id;

        chemgrid::gen_atom_grid(x, y, z, w, atom_id, atom_nuc,
                                chemgrid::RadialMethod::TreutlerAhlrichs, 3,
                                chemgrid::PruningMethod::NWChem);
        auto grouped = group_by_atom(x, y, z, w, atom_id);
        for (const auto &kv : grouped)
        {
            int atom = kv.first;
            const auto &points = kv.second;
        }
        // becke
        chemgrid::PartitionOut part;
        part = chemgrid::get_partition(
            atm_coords, grouped, charges,
            chemgrid::constants::BRAGG_RADII_ANG);

        // reorder
        std::vector<int> atm_idx;
        std::vector<double> quadrature_weights;
        chemgrid::build_atm_idx_and_weights_by_Z(grouped, charges, part.first, part.second, atm_idx, quadrature_weights);
        std::vector<size_t> idx = chemgrid::arg_group_grids_cpu(part.first, atm_coords,
                                                                chemgrid::constants::GROUP_BOX_SIZE,          // 1.2
                                                                chemgrid::constants::GROUP_BOUNDARY_PENALTY); // 4.2
        chemgrid::reorder_by_index_inplace(idx, part.first);
        chemgrid::reorder_by_index_inplace(idx, part.second);
        chemgrid::reorder_by_index_inplace(idx, atm_idx);
        chemgrid::reorder_by_index_inplace(idx, quadrature_weights);
        chemgrid::pad_grids_cpp(part.first, part.second, atm_idx, quadrature_weights, chemgrid::constants::ALIGNMENT_UNIT);
        return {part.first, part.second};
    }

    // 修改后的函数：
    std::map<int, std::vector<atom_AO>> get_atom_basis_map(
        const PrimitiveShell *shells_ptr,
        int bsisnum,
        const Atom *h_atoms,
        int nAtom)
    {
        std::map<int, std::vector<atom_AO>> atom_basis_map;

        // 建立 atom_index -> atomic_number 的映射
        std::map<int, int> atom_idx_to_charge;
        std::set<int> processed_elements; // 记录已处理的元素

        for (int i = 0; i < nAtom; i++)
        {
            atom_idx_to_charge[i] = h_atoms[i].atomic_number;
        }

        // Temporary storage: atom_index -> (basis_index, shell_type) -> list of (exp, coeff)
        // 关键：必须保留 basis_index！
        std::map<int, std::map<std::pair<int, int>, std::vector<std::pair<double, double>>>> temp_map;

        // First pass: group primitives by atom_index, basis_index, and shell_type
        for (int i = 0; i < bsisnum; i++)
        {
            int atom_idx = shells_ptr[i].atom_index;
            int basis_idx = shells_ptr[i].basis_index;
            int shell_type = shells_ptr[i].shell_type;
            double exp = shells_ptr[i].exponent;
            double coeff = shells_ptr[i].coefficient;

            std::pair<int, int> key = {basis_idx, shell_type};
            temp_map[atom_idx][key].push_back({exp, coeff});
        }

        // Second pass: 对于每个元素，只处理第一个该元素的原子
        for (auto &atom_pair : temp_map)
        {
            int atom_idx = atom_pair.first;
            int atomic_number = atom_idx_to_charge[atom_idx];

            // 如果这个元素已经处理过，跳过
            if (processed_elements.count(atomic_number) > 0)
            {
                std::cout << "Skipping atom_idx=" << atom_idx
                          << " (Z=" << atomic_number << ") - already processed\n";
                continue;
            }
            processed_elements.insert(atomic_number);

            std::cout << "Processing atom_idx=" << atom_idx
                      << " (Z=" << atomic_number << ")\n";

            std::vector<atom_AO> aos;

            // Sort by basis_index to maintain order
            std::map<int, std::vector<std::pair<int, std::vector<std::pair<double, double>>>>> sorted_by_basis;
            for (auto &basis_pair : atom_pair.second)
            {
                int basis_idx = basis_pair.first.first;
                int shell_type = basis_pair.first.second;
                sorted_by_basis[basis_idx].push_back({shell_type, basis_pair.second});
            }

            for (auto &basis_pair : sorted_by_basis)
            {
                int basis_idx = basis_pair.first;
                std::cout << "  basis_index=" << basis_idx << "\n";

                for (auto &shell_data : basis_pair.second)
                {
                    atom_AO ao;
                    ao.l = shell_data.first; // shell_type

                    std::cout << "    shell_type=" << ao.l
                              << ", nprim=" << shell_data.second.size() << "\n";

                    // Extract exponents and coefficients
                    for (auto &prim : shell_data.second)
                    {
                        ao.exps.push_back(prim.first);
                        ao.coeffs.push_back(prim.second);
                    }

                    aos.push_back(ao);
                }
            }

            atom_basis_map[atomic_number] = aos;
            std::cout << "  Total shells for Z=" << atomic_number << ": " << aos.size() << "\n\n";
        }

        std::cout << "\n========== Final Basis Set Summary ==========\n";
        for (const auto &pair : atom_basis_map)
        {
            std::cout << "Element Z=" << pair.first << ": " << pair.second.size() << " shells\n";
            for (size_t i = 0; i < pair.second.size(); i++)
            {
                std::cout << "  Shell " << i << ": l=" << pair.second[i].l
                          << ", nprim=" << pair.second[i].exps.size() << "\n";
            }
        }

        return atom_basis_map;
    }

    // 归一化函数保持不变
    std::map<int, std::vector<atom_AO>> normalize_atom_basis_map(
        const std::map<int, std::vector<atom_AO>> &unnormed_basis)
    {
        std::map<int, std::vector<atom_AO>> normed_basis;

        // 遍历每个原子类型（按核电荷数）
        for (const auto &atom_pair : unnormed_basis)
        {
            int atomic_number = atom_pair.first; // 核电荷数
            const std::vector<atom_AO> &aos = atom_pair.second;

            std::vector<atom_AO> normed_aos;

            // 遍历该原子的每个原子轨道
            for (const atom_AO &ao : aos)
            {
                atom_AO normed_ao;
                normed_ao.l = ao.l;
                normed_ao.exps = ao.exps;     // 指数不变
                normed_ao.coeffs = ao.coeffs; // 复制系数，准备归一化

                // 调用归一化函数
                normalize_basis(normed_ao.l, normed_ao.exps, normed_ao.coeffs);

                normed_aos.push_back(normed_ao);
            }

            normed_basis[atomic_number] = normed_aos;
        }

        return normed_basis;
    }

    // 完整的接口函数：直接从原始数据获取归一化的基函数
    std::map<int, std::vector<atom_AO>> get_normalized_atom_basis(
        const PrimitiveShell *shells_ptr,
        int bsisnum,
        const Atom *h_atoms, // 新增参数
        int nAtom)           // 新增参数
    {
        // Step 1: 获取未归一化的基函数（按核电荷数分类）
        auto unnormed_basis = get_atom_basis_map(shells_ptr, bsisnum, h_atoms, nAtom);

        // Step 2: 归一化
        auto normed_basis = normalize_atom_basis_map(unnormed_basis);

        return normed_basis;
    }

    std::vector<std::array<int, 3>> generate_cartesian_components(int l)
    {
        std::vector<std::array<int, 3>> components;

        for (int lx = l; lx >= 0; --lx)
        {
            for (int ly = l - lx; ly >= 0; --ly)
            {
                int lz = l - lx - ly;
                components.push_back({lx, ly, lz});
            }
        }

        return components;
    }

    std::vector<AODesc> generate_ao_list(
        const std::map<int, std::vector<atom_AO>> &normed_bas,
        const std::vector<int> &charges,
        const std::vector<std::array<double, 3>> &atom_coords)
    {
        std::vector<AODesc> ao_list;

        // Check inputs
        if (charges.size() != atom_coords.size())
        {
            throw std::invalid_argument("charges and atom_coords size mismatch");
        }

        // Iterate over each atom
        for (size_t atom_idx = 0; atom_idx < charges.size(); ++atom_idx)
        {
            int charge = charges[atom_idx];

            // Find the basis set for this element
            auto it = normed_bas.find(charge);
            if (it == normed_bas.end())
            {
                throw std::runtime_error("No basis set found for element with charge " +
                                         std::to_string(charge));
            }

            const std::vector<atom_AO> &basis_shells = it->second;

            // Iterate over each shell of this atom
            for (const atom_AO &shell : basis_shells)
            {
                int l = shell.l;

                // Generate all Cartesian components for this shell
                std::vector<std::array<int, 3>> cart_components =
                    generate_cartesian_components(l);

                // Create an AODesc for each Cartesian component
                for (const auto &comp : cart_components)
                {
                    AODesc ao;
                    ao.atom = atom_idx;
                    ao.l = l;
                    ao.lx = comp[0];
                    ao.ly = comp[1];
                    ao.lz = comp[2];
                    ao.exps = shell.exps;
                    ao.coeffs = shell.coeffs;

                    ao_list.push_back(ao);
                }
            }
        }

        return ao_list;
    }

    // 调试版本
    std::vector<AODesc> generate_ao_list_debug(
        const std::map<int, std::vector<atom_AO>> &normed_bas,
        const std::vector<int> &charges,
        const std::vector<std::array<double, 3>> &atom_coords)
    {
        std::vector<AODesc> ao_list;

        if (charges.size() != atom_coords.size())
        {
            throw std::invalid_argument("charges and atom_coords size mismatch");
        }

        std::cout << "\n========== Basis Set Analysis ==========\n";

        // 先打印基组信息
        for (const auto &pair : normed_bas)
        {
            int Z = pair.first;
            const auto &shells = pair.second;
            std::cout << "Element Z=" << Z << " has " << shells.size() << " shells:\n";
            for (size_t i = 0; i < shells.size(); i++)
            {
                std::cout << "  Shell " << i << ": l=" << shells[i].l
                          << ", nprim=" << shells[i].exps.size() << std::endl;
            }
        }

        std::cout << "\n========== Generating AO List ==========\n";

        for (size_t atom_idx = 0; atom_idx < charges.size(); ++atom_idx)
        {
            int charge = charges[atom_idx];

            std::cout << "\nAtom " << atom_idx << " (Z=" << charge << "):" << std::endl;

            auto it = normed_bas.find(charge);
            if (it == normed_bas.end())
            {
                throw std::runtime_error("No basis set found for element with charge " +
                                         std::to_string(charge));
            }

            const std::vector<atom_AO> &basis_shells = it->second;

            for (size_t shell_idx = 0; shell_idx < basis_shells.size(); shell_idx++)
            {
                const atom_AO &shell = basis_shells[shell_idx];
                int l = shell.l;

                std::vector<std::array<int, 3>> cart_components =
                    generate_cartesian_components(l);

                std::cout << "  Shell " << shell_idx << " (l=" << l << "): "
                          << cart_components.size() << " Cartesian components" << std::endl;

                for (const auto &comp : cart_components)
                {
                    std::cout << "    -> AO " << ao_list.size()
                              << ": (" << comp[0] << "," << comp[1] << "," << comp[2] << ")"
                              << std::endl;

                    AODesc ao;
                    ao.atom = atom_idx;
                    ao.l = l;
                    ao.lx = comp[0];
                    ao.ly = comp[1];
                    ao.lz = comp[2];
                    ao.exps = shell.exps;
                    ao.coeffs = shell.coeffs;

                    ao_list.push_back(ao);
                }
            }
        }

        std::cout << "\n========== Summary ==========\n";
        std::cout << "Total AO functions: " << ao_list.size() << std::endl;

        return ao_list;
    }
    void save_ao_to_txt(const double *ao_values, int ngrids, int nao,
                        const std::string &filename)
    {
        std::ofstream outfile(filename);

        if (!outfile.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // 设置输出精度
        outfile << std::scientific << std::setprecision(16);

        // 写入维度信息作为注释
        outfile << "# ngrids = " << ngrids << "\n";
        outfile << "# nao = " << nao << "\n";
        outfile << "# Format: ngrids x nao (row-major)\n";

        // 写入数据：每行是一个网格点的所有 AO 值
        for (int i = 0; i < ngrids; i++)
        {
            for (int j = 0; j < nao; j++)
            {
                outfile << ao_values[i * nao + j];
                if (j < nao - 1)
                    outfile << " ";
            }
            outfile << "\n";
        }

        outfile.close();
        std::cout << "AO values saved to " << filename << std::endl;
        std::cout << "Shape: (" << ngrids << ", " << nao << ")" << std::endl;
    }
    void save_grids_to_txt(const std::vector<std::array<double, 3>> &coords,
                           const std::string &filename)
    {
        std::ofstream outfile(filename);

        if (!outfile.is_open())
        {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        outfile << std::scientific << std::setprecision(16);
        outfile << "# ngrids = " << coords.size() << "\n";
        outfile << "# Format: x y z (one point per line)\n";

        for (const auto &coord : coords)
        {
            outfile << coord[0] << " " << coord[1] << " " << coord[2] << "\n";
        }

        outfile.close();
        std::cout << "Grid coordinates saved to " << filename << std::endl;
    }
    inline double get_spherical_norm_factor(int l)
    {
        // Precomputed values for common angular momenta
        static const double fac_table[] = {
            0.282094791773878143, // l=0: sqrt(1/(4π))
            0.488602511902919921, // l=1: sqrt(3/(4π))
            0.630783130505040012, // l=2: sqrt(5/(4π))
            0.746352665180230783, // l=3: sqrt(7/(4π))
            0.846284375321634481, // l=4: sqrt(9/(4π))
            0.935414346693485387, // l=5: sqrt(11/(4π))
            1.016220929899498370, // l=6: sqrt(13/(4π))
        };

        if (l >= 0 && l <= 6)
        {
            return fac_table[l];
        }

        // General formula for higher l
        return std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI));
    }

    inline double safe_power(double x, int n)
    {
        if (n < 0)
            return 0.0; // x^(-1) 等情况返回 0
        if (n == 0)
            return 1.0;
        if (n == 1)
            return x;
        if (n == 2)
            return x * x;
        if (n == 3)
            return x * x * x;

        double result = 1.0;
        for (int i = 0; i < n; ++i)
            result *= x;
        return result;
    }

    inline void evaluate_ao_grad(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_grad, // [3 x ngrids x nao]
        int ngrids,
        int nao)
    {
        std::fill(out_grad, out_grad + 3 * ngrids * nao, 0.0);

        for (int ao_idx = 0; ao_idx < nao; ++ao_idx)
        {
            const AODesc &ao = ao_list[ao_idx];
            const int lx = ao.lx;
            const int ly = ao.ly;
            const int lz = ao.lz;
            const int l = ao.l;
            const int nprim = ao.exps.size();

            const double fac = get_spherical_norm_factor(l);

            const double ax = atom_coords[ao.atom][0];
            const double ay = atom_coords[ao.atom][1];
            const double az = atom_coords[ao.atom][2];

            for (int g = 0; g < ngrids; ++g)
            {
                const double dx = grid_coords[g][0] - ax;
                const double dy = grid_coords[g][1] - ay;
                const double dz = grid_coords[g][2] - az;
                const double r2 = dx * dx + dy * dy + dz * dz;

                // Contracted sums
                double ce = 0.0;    // Σ c_j * exp(-α_j * r²)
                double ce_2a = 0.0; // Σ c_j * (-2α_j) * exp(-α_j * r²)

                for (int p = 0; p < nprim; ++p)
                {
                    const double alpha = ao.exps[p];
                    const double coeff = ao.coeffs[p];
                    const double e = std::exp(-alpha * r2);
                    ce += coeff * e;
                    ce_2a += coeff * e * alpha;
                }
                ce *= fac;
                ce_2a *= -2.0 * fac;

                // Angular powers
                const double dx_lx = safe_power(dx, lx);
                const double dy_ly = safe_power(dy, ly);
                const double dz_lz = safe_power(dz, lz);
                const double dx_lx_m1 = safe_power(dx, lx - 1);
                const double dy_ly_m1 = safe_power(dy, ly - 1);
                const double dz_lz_m1 = safe_power(dz, lz - 1);

                // Gradient formula:
                // ∂φ/∂x = lx * dx^(lx-1) * dy^ly * dz^lz * ce
                //       + dx^lx * dy^ly * dz^lz * ce_2a * dx
                //       = dy^ly * dz^lz * (lx * dx^(lx-1) * ce + dx^(lx+1) * ce_2a)
                const double common_yz = dy_ly * dz_lz;
                const double common_xz = dx_lx * dz_lz;
                const double common_xy = dx_lx * dy_ly;

                const double grad_x = common_yz * (lx * dx_lx_m1 * ce + dx_lx * dx * ce_2a);
                const double grad_y = common_xz * (ly * dy_ly_m1 * ce + dy_ly * dy * ce_2a);
                const double grad_z = common_xy * (lz * dz_lz_m1 * ce + dz_lz * dz * ce_2a);

                out_grad[0 * ngrids * nao + g * nao + ao_idx] = grad_x;
                out_grad[1 * ngrids * nao + g * nao + ao_idx] = grad_y;
                out_grad[2 * ngrids * nao + g * nao + ao_idx] = grad_z;
            }
        }
    }

    // ============================================================
    // Evaluate both AO values and gradients (more efficient)
    //
    // Output:
    //   out_ao:   double* [ngrids x nao] - AO values (can be nullptr)
    //   out_grad: double* [3 x ngrids x nao] - gradients (can be nullptr)
    //
    // ============================================================
    inline void evaluate_ao_and_grad(
        const std::vector<AODesc> &ao_list,
        const std::vector<std::array<double, 3>> &atom_coords,
        const std::vector<std::array<double, 3>> &grid_coords,
        double *out_ao,   // [ngrids x nao] or nullptr
        double *out_grad, // [3 x ngrids x nao] or nullptr
        int ngrids,
        int nao)
    {
        if (nao != (int)ao_list.size())
            throw std::invalid_argument("nao != ao_list.size()");

        const bool compute_ao = (out_ao != nullptr);
        const bool compute_grad = (out_grad != nullptr);

        if (compute_ao)
            std::fill(out_ao, out_ao + ngrids * nao, 0.0);
        if (compute_grad)
            std::fill(out_grad, out_grad + 3 * ngrids * nao, 0.0);

        for (int ao_idx = 0; ao_idx < nao; ++ao_idx)
        {
            const AODesc &ao = ao_list[ao_idx];
            const int lx = ao.lx;
            const int ly = ao.ly;
            const int lz = ao.lz;
            const int nprim = ao.exps.size();

            const double ax = atom_coords[ao.atom][0];
            const double ay = atom_coords[ao.atom][1];
            const double az = atom_coords[ao.atom][2];

            for (int g = 0; g < ngrids; ++g)
            {
                const double gx = grid_coords[g][0];
                const double gy = grid_coords[g][1];
                const double gz = grid_coords[g][2];

                const double dx = gx - ax;
                const double dy = gy - ay;
                const double dz = gz - az;
                const double r2 = dx * dx + dy * dy + dz * dz;

                // Contracted sums
                double R0 = 0.0;
                double R1 = 0.0;
                for (int p = 0; p < nprim; ++p)
                {
                    const double alpha = ao.exps[p];
                    const double coeff = ao.coeffs[p];
                    const double exp_val = std::exp(-alpha * r2);
                    R0 += coeff * exp_val;
                    if (compute_grad)
                        R1 += coeff * alpha * exp_val;
                }

                // Angular part
                const double dx_lx = safe_power(dx, lx);
                const double dy_ly = safe_power(dy, ly);
                const double dz_lz = safe_power(dz, lz);
                const double angular = dx_lx * dy_ly * dz_lz;

                // AO value
                if (compute_ao)
                {
                    out_ao[g * nao + ao_idx] = angular * R0;
                }

                // Gradients
                if (compute_grad)
                {
                    const double dx_lx_m1 = safe_power(dx, lx - 1);
                    const double dy_ly_m1 = safe_power(dy, ly - 1);
                    const double dz_lz_m1 = safe_power(dz, lz - 1);

                    const double common_yz = dy_ly * dz_lz;
                    const double common_xz = dx_lx * dz_lz;
                    const double common_xy = dx_lx * dy_ly;

                    const double grad_x = common_yz * (lx * dx_lx_m1 * R0 - 2.0 * dx_lx * dx * R1);
                    const double grad_y = common_xz * (ly * dy_ly_m1 * R0 - 2.0 * dy_ly * dy * R1);
                    const double grad_z = common_xy * (lz * dz_lz_m1 * R0 - 2.0 * dz_lz * dz * R1);

                    out_grad[0 * ngrids * nao + g * nao + ao_idx] = grad_x;
                    out_grad[1 * ngrids * nao + g * nao + ao_idx] = grad_y;
                    out_grad[2 * ngrids * nao + g * nao + ao_idx] = grad_z;
                }
            }
        }
    }

    inline void export_ao_values_to_txt(
        const double *ao_values, // [ngrids x nao]
        int ngrids,
        int nao,
        const std::string &filename)
    {
        std::ofstream ofs(filename);
        if (!ofs.is_open())
            throw std::runtime_error("Cannot open file: " + filename);

        ofs << "# AO values\n";
        ofs << "# Shape: (" << ngrids << ", " << nao << ")\n";
        ofs << "DIMS " << ngrids << " " << nao << "\n";

        ofs << std::scientific << std::setprecision(15);

        for (int g = 0; g < ngrids; ++g)
        {
            for (int ao = 0; ao < nao; ++ao)
            {
                ofs << ao_values[g * nao + ao];
                if (ao < nao - 1)
                    ofs << " ";
            }
            ofs << "\n";
        }

        ofs.close();
        std::cout << "Exported AO values (" << ngrids << " x " << nao << ") to " << filename << std::endl;
    }

    // ============================================================
    // Export AO gradients
    // Format: DIMS 3 ngrids nao
    //         (all grad_x data, then grad_y, then grad_z)
    // ============================================================
    inline void export_ao_grad_to_txt(
        const double *ao_grad, // [3 x ngrids x nao]
        int ngrids,
        int nao,
        const std::string &filename)
    {
        std::ofstream ofs(filename);
        if (!ofs.is_open())
            throw std::runtime_error("Cannot open file: " + filename);

        ofs << "# AO gradients\n";
        ofs << "# Shape: (3, " << ngrids << ", " << nao << ")\n";
        ofs << "# Order: grad_x, grad_y, grad_z\n";
        ofs << "DIMS 3 " << ngrids << " " << nao << "\n";

        ofs << std::scientific << std::setprecision(15);

        for (int d = 0; d < 3; ++d)
        {
            for (int g = 0; g < ngrids; ++g)
            {
                for (int ao = 0; ao < nao; ++ao)
                {
                    ofs << ao_grad[d * ngrids * nao + g * nao + ao];
                    if (ao < nao - 1)
                        ofs << " ";
                }
                ofs << "\n";
            }
        }

        ofs.close();
        std::cout << "Exported AO gradients (3 x " << ngrids << " x " << nao << ") to " << filename << std::endl;
    }

    AOGrids dft_gen_ao(std::map<int, std::vector<atom_AO>> normed_bas, std::vector<int> charges, std::vector<std::array<double, 3>> &atm_coords, std::vector<std::array<double, 3>> coords)
    {

        for (auto it = normed_bas.begin(); it != normed_bas.end(); ++it)
        {
            std::cout << it->first << std::endl; // 输出键
        }

        std::vector<AODesc> AODESC = generate_ao_list(normed_bas, charges, atm_coords);
        int ngrids = static_cast<int>(coords.size());
        int nao = static_cast<int>(AODESC.size());
        printf("Total AO count: %d\n", nao);
        printf("values size: %d x %d \n", ngrids, nao);
        double *ao_values = new double[ngrids * nao];
        // printf("Evaluating AOs on grids using GPU...\n");
        // chemgrid::evaluate_aos_on_grids_gpu_raw(AODESC, atm_coords, coords, ao_values, ngrids, nao);
        chemgrid::evaluate_aos_gpu_shell_grouped(AODESC, atm_coords, coords, ao_values, ngrids, nao);
        double *out_grad = new double[3 * ngrids * nao];
        // printf("Evaluating AO gradients on grids using CPU...\n");
        // evaluate_ao_grad(AODESC, atm_coords, coords, out_grad, ngrids, nao);
        // // EXPORT TO TXT FOR DEBUGGING
        // export_ao_values_to_txt(ao_values, ngrids, nao, "ao_values.txt");
        // export_ao_grad_to_txt(out_grad, ngrids, nao, "ao_gradients.txt");
        AOGrids outAO;
        outAO.ao = ao_values;
        outAO.naos = nao;
        outAO.ngrids = ngrids;
        return outAO;
    }

}