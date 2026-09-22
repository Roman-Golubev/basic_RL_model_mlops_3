import math
import pandas as pd
import numpy as np

from scripts.properties import (
    row, nuw, cpw, lbdw,
    lbdex, nuex, cpex, Friclam
)


def RL_func(
    init_param_row, fins_df, alpha, gamma, epsilon_now, seed,
    total_reward, steps_total, Pus_prev, Q, tem_distr, fin_num,
    s, Pus_final
):
    rng = np.random.default_rng(seed)
    cur_row = init_param_row.copy()
    Pus_0 = cur_row['Pus_full']
    fin_type = cur_row['fin_type']
    fin_num_0 = fins_df[fins_df['fin_type'] == fin_type].index[0]
    tem_distr_0 = np.array([100] * 10 * 3).reshape(3, 10)

    states_lst = []
    actions_lst = []
    rewards_lst = []
    del_Pus_lst = []
    tem_distr_final = np.array([-1] * 10 * 3).reshape(3, 10)
    fin_num_final = -1
    step_num_final = -1
    for step in range(50):
        # выбор действия
        if rng.random() < epsilon_now:
            a = int(rng.integers(Q.shape[1]))
        else:
            a = int(np.argmax(Q[s]))

        # выполнение шага
        if a <= 59:
            incr = a%2 == 0
            idx = a//2
            idx_arr = np.zeros(30)
            idx_arr[idx] = 1
            idx_arr = idx_arr.reshape(3, 10).astype(bool)
            cur_param = tem_distr[idx_arr][0]
            if incr:
                if cur_param < 100:
                    tem_distr[idx_arr] += 10
            else:
                if cur_param >= 40:
                    tem_distr[idx_arr] -= 10
        else:
            if a == 60:
                if fin_num < 8:
                    fin_num += 1
            else:
                if fin_num > 0:
                    fin_num -= 1
            cur_row['fin_type'] = fins_df.loc[fin_num, 'fin_type']
            cur_row['hhm'] = fins_df.loc[fin_num, 'hhm']
            cur_row['dekvhm'] = fins_df.loc[fin_num, 'dekvhm']
            cur_row['shaghm'] = fins_df.loc[fin_num, 'shaghm']
            cur_row['deltahm'] = fins_df.loc[fin_num, 'deltahm']
            cur_row['ledhm'] = fins_df.loc[fin_num, 'ledhm']
        results_dict = rl_model(cur_row, tem_distr, full_design=False)
        if results_dict['message'] == 'Расчёт выполнен':
            Pus_cur = results_dict['Pus']
            delta_Pus = (Pus_cur - Pus_0) * 100 / Pus_0
            if delta_Pus <= -15:
                # возврат в исходно состояние
                s_next, r, done = 0, -50, False
                tem_distr = tem_distr_0.copy()
                fin_num = fin_num_0
                Pus_prev = 0
            if delta_Pus > -15 and delta_Pus <= 0:
                Pus_prev = Pus_cur
                s_next, r, done = 0, -1, False
            if delta_Pus > 0 and delta_Pus <= 5:
                r = -1 if Pus_cur == Pus_prev else 1
                Pus_prev = Pus_cur
                s_next, done = 1, False
                # сохранение результата
                if Pus_cur > Pus_final:
                    Pus_final = Pus_cur
                    tem_distr_final = tem_distr.copy()
                    fin_num_final = fin_num
                    step_num_final = step
            if delta_Pus > 5 and delta_Pus <= 15:
                r = -1 if Pus_cur == Pus_prev else 5
                Pus_prev = Pus_cur
                s_next, done = 2, False
                # сохранение результата
                if Pus_cur > Pus_final:
                    Pus_final = Pus_cur
                    tem_distr_final = tem_distr.copy()
                    fin_num_final = fin_num
                    step_num_final = step
            if delta_Pus > 15:
                s_next, r, done = 2, 50, True
                # сохранение результата
                Pus_final = Pus_cur
                tem_distr_final = tem_distr.copy()
                fin_num_final = fin_num
                step_num_final = step
        else:
            # возврат в исходно состояние
            s_next, r, done = 0, -50, False
            tem_distr = tem_distr_0.copy()
            fin_num = fin_num_0
            Pus_prev = 0
            delta_Pus = -100

        states_lst.append(s)
        actions_lst.append(a)
        rewards_lst.append(r)
        del_Pus_lst.append(delta_Pus)

        # Обновление Q-значений
        total_reward += r
        target = r if done else r + gamma * np.max(Q[s_next])
        Q[s, a] += alpha * (target - Q[s, a])
        s = s_next
        if done:
            break

    steps_total += 50
    return (
        Q, tem_distr, fin_num, Pus_cur, total_reward, steps_total, s, done,
        states_lst, actions_lst, rewards_lst, del_Pus_lst,
        Pus_final, tem_distr_final, fin_num_final, step_num_final
    )


def layer_distr_func(distr_perc, rownumtot, nTEMrow, Lpolez):
    row_indices = np.array(list(range(1, rownumtot+1)))
    row_indices = np.round((row_indices - 1)*9/(rownumtot - 1)).astype(int)
    distr_perc_cur = distr_perc[row_indices]
    TEM_cnt_lst = []
    Lpolez1, TEM_qua_cur = 1, nTEMrow
    while Lpolez1 > Lpolez:
        Lpolez1 = TEM_qua_cur/nTEMrow
        if Lpolez1 >= Lpolez:
            TEM_cnt_lst.append(TEM_qua_cur)
            TEM_qua_cur -= 2
    TEM_cnt_min = min(TEM_cnt_lst)
    TEM_cnt = np.array([nTEMrow]*rownumtot)
    TEM_cnt = np.round(TEM_cnt*distr_perc_cur/100).astype(int)
    TEM_cnt[TEM_cnt < TEM_cnt_min] = TEM_cnt_min
    TEM_cnt[~np.isin(TEM_cnt, TEM_cnt_lst)] += 1
    layer_distr = np.full((rownumtot + 1, nTEMrow + 1), 1)
    for row_idx in range(rownumtot):
        # распределение ТЭМ
        if TEM_cnt[row_idx] < nTEMrow:
            pustsl = nTEMrow - TEM_cnt[row_idx]
            nzaz = TEM_cnt[row_idx] - 1
            lzazch = pustsl / nzaz - math.trunc(pustsl / nzaz)
            lzaz2 = math.trunc(pustsl / nzaz) if lzazch == 0 else math.trunc(pustsl / nzaz) + 1  # длина большого промежутка
            lzaz1 = lzaz2 - 1  # длина малого промежутка
            yzaz = pustsl - nzaz * lzaz1  # количество больших промежутков
            xzaz = nzaz - yzaz  # количество малых промежутков
            sl, s3, sim, sim1 = 1, 0, 0, 0
            xzaz2 = math.trunc(xzaz / 2)
            xzaz3 = xzaz / 2 - xzaz2
            yzaz2 = math.trunc(yzaz / 2)
            yzaz3 = yzaz / 2 - yzaz2
            if yzaz > 1:
                yon = 1
                xon = 0
            else:
                yon = 0
                xon = 1

            # заполнение слотов
            for jm in range(1, nTEMrow + 1):
                # до центра
                if sim == 0:
                    # есть ТЭМ
                    if sl == 1:
                        layer_distr[row_idx+1, jm] = 1
                        if yon == 1:
                            sl = 0
                        if xon == 1:
                            if lzaz1 > 0:
                                sl = 0
                            # заполнение для меньших промежутков нулевой длины
                            else:
                                masparam = layer_distr[row_idx+1, jm - 1]
                                if masparam == 1:
                                    xzaz2 -= 1
                        if xzaz2 == 0 and yzaz2 == 0:
                            sim = 1
                    # нет ТЭМ
                    else:
                        layer_distr[row_idx+1, jm] = 0
                        # по большему промежутку
                        if yon == 1:
                            lzaz2 -= 1
                            if lzaz2 == 0:
                                lzaz2 = math.trunc(pustsl / nzaz) if lzazch == 0 else math.trunc(pustsl / nzaz) + 1
                                sl = 1
                                yzaz2 -= 1
                            if yzaz2 == 0:
                                yon = 0
                                if xzaz2 > 0:
                                    xon = 1
                        # по меньшему промежутку
                        else:
                            lzaz1 -= 1
                            if lzaz1 == 0:
                                lzaz1 = lzaz2 - 1
                                sl = 1
                                xzaz2 -= 1
                        if xzaz2 == 0 and yzaz2 == 0 and xzaz3 == 0 and yzaz3 == 0: # новое
                            sim = 1

                # начиная с центральных слотов
                else:
                    # центральные слоты
                    if sim1 == 0:
                        # в центре больший промежуток
                        if yzaz3 > 0:
                            if lzaz2 > 0:
                                layer_distr[row_idx+1, jm] = 0
                                lzaz2 -= 1
                            else:
                                sim1 = 1
                                layer_distr[row_idx+1, jm] = 1
                        # в центре меньший промежуток
                        if xzaz3 > 0:
                            if lzaz1 > 0:
                                layer_distr[row_idx+1, jm] = 0
                                lzaz1 -= 1
                            else:
                                sim1 = 1
                                layer_distr[row_idx+1, jm] = 1
                        # ТЭМ на оси симметрии
                        if xzaz3 == 0 and yzaz3 == 0:
                            sim1 = 1
                            layer_distr[row_idx+1, jm] = 1
                    # за центром
                    else:
                        layer_distr[row_idx+1, nTEMrow - s3] = layer_distr[row_idx+1, s3 + 1]
                        s3 += 1
    TEM_cnt = np.hstack([nTEMrow, TEM_cnt])
    return layer_distr, TEM_cnt


def rl_model(cur_row: pd.Series, tem_distr: np.ndarray, full_design=True):
    (
        TEM_type, Tchm0, Tccm0, N0, I0, eta0, aTEM, aTE, Ncoup, hTEM, mTEM,
        prop, fin_type, hhm, deltahm, shaghm, dekvhm, ledhm, lbdNTEG, roNTEG,
        lbdOTEG, roOTEG, Lpolez, nTEMrowmax, rowmax, wbefdif, Thm0, Thmpr,
        wcment, Tcm0, Tcmpr, rtchmmax, rtcwmax, d, Ne, alphaGD, gDT
    ) = (
        cur_row['TEM_type'], cur_row['Tchm0'], cur_row['Tccm0'], cur_row['N0'],
        cur_row['I0'], cur_row['eta0'], cur_row['aTEM'], cur_row['aTE'],
        cur_row['Ncoup'], cur_row['hTEM'], cur_row['mTEM'], cur_row['prop'],
        cur_row['fin_type'], cur_row['hhm'], cur_row['deltahm'], cur_row['shaghm'],
        cur_row['dekvhm'], cur_row['ledhm'], cur_row['lbdNTEG'], cur_row['roNTEG'],
        cur_row['lbdOTEG'], cur_row['roOTEG'], cur_row['Lpolez'], cur_row['nTEMrowmax'],
        cur_row['rowmax'], cur_row['wbefdif'], cur_row['Thm0'], cur_row['Thmpr'],
        cur_row['wcment'], cur_row['Tcm0'], cur_row['Tcmpr'], cur_row['rtchmmax'],
        cur_row['rtcwmax'], cur_row['d'], cur_row['Ne'], cur_row['alphaGD'], cur_row['gDT']
    )

    mprop = np.zeros(20)
    mThmin = np.zeros(10000)
    mThmout = np.zeros(10000)
    mTcmout = np.zeros(10000)
    mwhmout = np.zeros(10000)

    # Словарь результатов расчёта
    results_dict = {
        'TEM_type': TEM_type, 'Tchm0': Tchm0, 'Tccm0': Tccm0, 'N0': N0, 'I0': I0, 'eta0': eta0, 'aTEM': aTEM, 'aTE': aTE, 'Ncoup': Ncoup, 'hTEM': hTEM, 'mTEM': mTEM,
        'prop': prop, 'fin_type': fin_type, 'hhm': hhm, 'deltahm': deltahm, 'shaghm': shaghm, 'dekvhm': dekvhm, 'ledhm': ledhm, 'lbdNTEG': lbdNTEG, 'roNTEG': roNTEG,
        'lbdOTEG': lbdOTEG, 'roOTEG': roOTEG, 'Lpolez': Lpolez, 'nTEMrowmax': nTEMrowmax, 'rowmax': rowmax, 'wbefdif': wbefdif, 'Thm0': Thm0, 'Thmpr': Thmpr,
        'wcment': wcment, 'Tcm0': Tcm0, 'Tcmpr': Tcmpr, 'rtchmmax': rtchmmax, 'rtcwmax': rtcwmax, 'd': d, 'Ne': Ne, 'alphaGD': alphaGD, 'gDT': gDT,
        'message': None, 'TEM_distrib_TEG': None, 'Tcm': None, 'Thm': None, 'Pelem_arr': None,
        'Qhm_arr': None, 'Rhm_arr': None, 'Rcm_arr': None, 'Tchm_arr': None, 'Tccm_arr': None,
        'widthhm': None, 'BTEG': None, 'widthcm': None, 'LTEG': None, 'heightsum': None,
        'mTEG': None, 'mTEG_empty': None, 'nNTEG': None, 'cmflownum': None, 'cmcannummax': None,
        'cmcannummin': None, 'cmgrmax': None, 'TEMlayq2': None, 'rownumtot1': None, 'nTEMrowhm': None,
        'UA': None, 'NTU': None, 'eps': None, 'Rhm': None, 'Rcm': None,
        'kT': None, 'Qhm': None, 'qhmplot': None, 'Tchm': None, 'Tccm': None,
        'deltaTc': None, 'eta': None, 'Pelem': None, 'PTEG': None, 'Nrtccm': None,
        'Pus': None, 'Arthm_HE': None, 'Arthm': None, 'massflow_hm_HE': None, 'massflow_hm': None,
        'rtchmap': None, 'rtchm': None, 'whm': None, 'ThmTEGout': None, 'temkGhm': None,
        'alphateplfhm': None, 'Artcm_HE': None, 'Artcm': None, 'massflow_cm_HE': None, 'massflow_cm': None,
        'rtccmap': None, 'rtccm': None, 'wcm': None, 'TcmTEGout': None, 'temkGcm': None, 'alphateplfcm': None
    }

    # Дефолтные настроечные коэффициенты
    kiz = 0.98
    kkonvhm = 0.93
    kkonvcm = 0.93
    ktp = 1.04
    knagr = 1.15
    kro = 0.91
    kalpha = 0.99
    klbd = 1.01

    nTE = Ncoup*2
    ArTE = aTE**2
    Arcer = aTEM**2

    ro0 = kro*N0*ArTE/(nTE*I0**2*hTEM)
    alpha0 = 2*kalpha*N0/(nTE*(Tchm0 - Tccm0)*I0)
    lbd0 = klbd*((
        N0*hTEM/(ArTE*nTE*(Tchm0 - Tccm0)*eta0)
    ) - (
        alpha0**2*(((Tchm0 + Tccm0)/2) + Tchm0)/(4*ro0)
    ))
    Z0 = alpha0**2/(ro0*lbd0)
    R = hTEM/lbd0
    m = knagr

    CNG, HNG, NNG, MNG, dCNG, dHNG, QRNNG, gNG = 0, 0, 0, 0, 0, 0, 0, 0
    gsum = gDT + gNG
    dCmix = ((gNG/gsum)*dCNG + (gDT/gsum)*0.87)*100
    dHmix = ((gNG/gsum)*dHNG + (gDT/gsum)*0.126)*100
    L0 = 0.115*dCmix + 0.345*dHmix - 0.043*(gDT/gsum)*0.004*100
    GT = gsum*Ne/3600
    Gex = GT*(1 + alphaGD*L0)
    V0 = 0.0889*dCmix + 0.267*dHmix - 0.0333*(gDT/gsum)*0.004*100
    VRO2 = 0.0187*dCmix
    VH2O = 0.111*dHmix + 1.6*d*alphaGD*V0
    V0H2O = 0.111*dHmix + 1.6*d*V0
    V0N2 = 0.79*V0
    V0G = VRO2 + V0N2 + V0H2O
    VG = V0G + (1 + 1.6*d)*(alphaGD-1)*V0
    rH2O = VH2O/VG
    Vhm = GT*VG*Thm0/273
    rohm0 = Gex/Vhm

    rocm0 = row(Tcm0)

    comprdiam_dict = {
        0: [0.006, 0.008, 0.010, 0.012],
        1: [0.008, 0.010, 0.012, 0.016],
        2: [0.010, 0.012, 0.013, 0.020],
        3: [0.012, 0.014, 0.014, 0.024],
        4: [0.014, 0.016, 0.016, 0.028],
        5: [0.016, 0.018, 0.017, 0.030],
    }
    calccomprdiam, comprch, axcomprtype, calccontin = 0, 0, 0, 1
    # крепёж стяжки без промежуточных шпилек
    while comprch == 0:
        comprdiam, dtr, deltaTEM, diambob = comprdiam_dict[calccomprdiam]
        axdist = aTEM + deltaTEM
        axdistmax = 5*comprdiam;
        if axdist <= axdistmax:
           comprch = 1
           axcomprtype = 1
        else:
            calccomprdiam += 1
        if calccomprdiam > 5:
            comprch = 1
    # одна промежуточная шпилька
    if calccomprdiam > 5:
        calccomprdiam, comprch = 0, 0
        while comprch == 0:
            comprdiam, dtr, deltaTEM, diambob = comprdiam_dict[calccomprdiam]
            axdist = (aTEM + deltaTEM)/2
            axdistmax = 5*comprdiam;
            if axdist <= axdistmax:
                comprch = 1
                axcomprtype = 2
            else:
                calccomprdiam += 1
            if calccomprdiam > 5:
                comprch = 1
    # две промежуточная шпилька
    if calccomprdiam > 5:
        calccomprdiam, comprch = 0, 0
        while comprch == 0:
            comprdiam, dtr, deltaTEM, diambob = comprdiam_dict[calccomprdiam]
            axdist = (aTEM + deltaTEM)/3
            axdistmax = 5*comprdiam;
            if axdist <= axdistmax:
                comprch = 1
                axcomprtype = 3
            else:
                calccomprdiam += 1
            if calccomprdiam > 5:
                comprch = 1

    if axcomprtype == 1:
        deltaedg = (deltaTEM + diambob)/2
    if axcomprtype == 2 or axcomprtype == 3:
        deltaedg = (dtr + diambob)/2 + 0.003
    if axcomprtype == 0:
        calccontin = 0

    # подобран осевой крепёж
    if calccontin == 1:
        hcms = 0.005
        hcmm = 0.010
        deltacm = 0.001
        bsthichm = 0.001
        bsthiccm = 0.002
        HOTEGm = hcmm + 2*bsthiccm + 2*(hTEM - 0.001)
        HNTEG = hhm + 2*bsthichm

        fldiam_dict = {
            0: [0.006, 0.024, 0.013, 0.019, 0.012],
            1: [0.008, 0.028, 0.018, 0.025, 0.016],
            2: [0.010, 0.032, 0.023, 0.030, 0.020],
            3: [0.012, 0.036, 0.029, 0.036, 0.024],
            4: [0.014, 0.040, 0.034, 0.041, 0.028],
            5: [0.016, 0.042, 0.039, 0.045, 0.030],
        }
        calcfldiam, flch = 0, 0
        while flch == 0:
            fldiam, aflb, acutmin, bflb, Hwheetmin = fldiam_dict[calcfldiam]
            if HNTEG < Hwheetmin:
                flch = 1
                calccontin = 0
            else:
                HOTEGs = hcms + 2*bsthiccm + (hTEM - 0.001) + Hwheetmin + 0.005
                if HOTEGm < Hwheetmin:
                    HOTEGm = Hwheetmin
                    hcmm = HOTEGm - 2*bsthiccm - 2*(hTEM - 0.001)
                    hcms = hcmm/2
                    HOTEGs = hcms + 2*bsthiccm + (hTEM - 0.001) + Hwheetmin + 0.005
                fldist = HOTEGm/2 + 0.001 + HNTEG/2
                fldistmax = 5*fldiam
                if fldist <= fldistmax:
                    flch = 1
                else:
                    calcfldiam = calcfldiam + 1
                if calcfldiam > 5:
                    flch = 1
        if calcfldiam > 5:
            calccontin = 0

        # фланцевые бобышки подобраны
        if calccontin == 1:
            shagcm = hcms
            if fin_type != 'Continuous':
                dhidrhm = dekvhm
                ReyNus = 3960 * (deltahm/dhidrhm)**0.25 * (ledhm/dhidrhm)**0.42
                etafhmtot = 1
            whm0 = 35 if wbefdif > 35 else wbefdif
            uslwexmax = whm0*1000
            uslwex0 = whm0*1000
            rowmax1 = rowmax
            rtcsuc, delrownumtot, nextdrow = 0, 1, 0

            # расчёт гидравлического сопротивления для горячей среды
            while rtcsuc == 0:
                hmflownum, TEMnumsuc = 1, 0
                # предельное количество ТЭМ в ряду
                while TEMnumsuc == 0:
                    propsuc, dprop0, nNTEG, flbdist, thickst, deltab, distcov = (
                        0, 1000000, hmflownum, 0, 0, 0, 0
                    )
                    # соотношение сторон ТЭГ
                    while propsuc == 0:
                        lw0, nTEMrowhm = 0, 4
                        # действительная скорость на входе
                        while lw0 == 0:
                            Lhm = aTEM*nTEMrowhm + deltaTEM*(nTEMrowhm - 1) + 2*deltaedg
                            Arrowhm = (hhm - deltahm)*(shaghm - deltahm)*Lhm*nNTEG/shaghm
                            whm01 = Vhm/Arrowhm
                            uslwex01 = round(whm01*1000)
                            if uslwex01 > uslwex0:
                                nTEMrowhm = nTEMrowhm + 1
                            else:
                                lw0 = 1
                        whm0 = whm01
                        widthhm0 = aTEM*nTEMrowhm + deltaTEM*(nTEMrowhm - 1) + 2*(deltaedg - (diambob - (dtr + 0.006))/2) + 0.040
                        thickst = (widthhm0 - Lhm)/2
                        flbdist = (widthhm0 - (aTEM*nTEMrowhm + deltaTEM*(nTEMrowhm - 1)))/2
                        distcov = aflb + acutmin - flbdist
                        # наложение следа бобышки на слой ТЭМ
                        if distcov > thickst:
                            deltab = (aflb + acutmin - thickst - flbdist)/2
                            widthhm = widthhm0 + 2*deltab
                            thickst = (widthhm - Lhm)/2
                            flbdist = flbdist + deltab
                        else: widthhm = widthhm0 # нет наложения следа бобышки на слой ТЭМ
                        heightsum = HNTEG*nNTEG + HOTEGm*(nNTEG - 1) + 2*HOTEGs + 0.001*2*nNTEG
                        prop1 = widthhm/heightsum
                        dprop = abs(prop - prop1)
                        if dprop0 < dprop:
                            nNTEG, Lhm, widthhm, widthhm0, heightsum, prop1, flbdist, nTEMrowhm1 = (
                                mprop[0], mprop[1], mprop[2], mprop[3], mprop[4], mprop[5], mprop[6], mprop[7]
                            )
                            nTEMrowhm, thickst, deltab, distcov, whm0, Arrowhm = (
                                round(nTEMrowhm1), mprop[8], mprop[9], mprop[10], mprop[11], mprop[12]
                            )
                            propsuc = 1
                        else:
                            (
                                mprop[0], mprop[1], mprop[2], mprop[3], mprop[4], mprop[5], mprop[6],
                                mprop[7], mprop[8], mprop[9], mprop[10], mprop[11], mprop[12]
                            ) = (
                                nNTEG, Lhm, widthhm, widthhm0, heightsum, prop1, flbdist,
                                nTEMrowhm, thickst, deltab, distcov, whm0, Arrowhm
                            )
                            nNTEG = nNTEG + 1
                            dprop0 = dprop
                    if nTEMrowhm <= nTEMrowmax:
                        TEMnumsuc = 1
                    else: # количество ТЭМ в ряду больше предельного
                        TEMnumsuc = 1
                        if uslwex0 < uslwexmax:
                            nextdrow = 1
                        else:
                            calccontin = 0
                # превышение количества ТЭМ не произошло
                if nextdrow == 0 and calccontin == 1:
                    hmflownum, hmcannummax, hmcannummin = round(nNTEG), 1, 0

                    dpatrhm = 2 * (Vhm/(3.14*wbefdif))**0.5
                    rgib = (HOTEGm + 0.002 + HNTEG)/2
                    hobt = (HOTEGm + 0.004)/0.5359
                    acol1 = heightsum - 2*HOTEGs - 0.004
                    acol2 = widthhm - 2*aflb
                    acol3 = acol1 + 0.5359*hobt
                    dF1difhm = 3.14 * dpatrhm**2 / (4*acol2*acol3)
                    dF2difhm = nNTEG*acol2*hhm/(acol2*acol3)
                    dF3difhm = Arrowhm/(nNTEG*acol2*hhm)
                    rtcpatrent = (1 / (2*math.log(dpatrhm/0.0004 , 10) + 1.14)**2)*wbefdif**2*rohm0/4
                    rtcdif = (1.183*dF1difhm**2 - 2.3038*dF1difhm + 1.1259)*1*wbefdif**2*rohm0/2
                    waftdif = wbefdif*dF1difhm
                    dpatrper = 4*acol2*acol3/(2*(acol2 + acol3))
                    rtcpatrperent = (1 / (2*math.log(dpatrper/0.0004 , 10) + 1.14)**2)*(dpatrhm/(2*dpatrper))*waftdif**2*rohm0/2
                    waftobtent = waftdif/dF2difhm
                    rtcdifobt = 0.1*waftobtent**2*rohm0/2
                    rtcdifper = (-0.494*dF3difhm + 0.497)*whm0**2*rohm0/2

                    wcm0 = wcment
                    wcmin = wcm0
                    whmin = whm0
                    Tcmin = Tcm0
                    Thmin = Thm0
                    teplsuc = 0
                    imas, rownum, rownumtot, rowf, PTEG = 1, 1, 4, 0, 0
                    TEMnum = nTEMrowhm

                    # тепловой расчёт рядами по ОГ
                    while teplsuc == 0:
                        Lcm = aTEM*rownumtot + deltaTEM*(rownumtot - 1) + 2*deltaedg
                        Arfhm = nNTEG*math.trunc(Lhm/shaghm)*2*(hhm - deltahm)*Lcm
                        Arfcm = math.trunc(Lcm/shagcm)*2*Lhm*((nNTEG - 1)*(hcmm - deltacm) + 2*(hcms - deltacm))
                        if fin_type == 'Continuous':
                            Arthm = 2*Lhm*Lcm*nNTEG - 2*deltahm*Lcm*(Lhm*nNTEG/shaghm) + \
                                2*hhm*Lcm*nNTEG + 2*(hhm - deltahm)*Lcm*(Lhm*nNTEG/shaghm) + \
                                2*(hhm - deltahm)*deltahm*(Lhm*nNTEG/shaghm) + 2*shaghm*deltahm*(Lhm*nNTEG/shaghm)
                            dhidrhm = 4*Arrowhm*Lcm/Arthm
                        else:
                            Arthm = 2*Lhm*Lcm*nNTEG - 2*deltahm*Lcm*(Lhm*nNTEG/shaghm) + 2*hhm*Lcm*nNTEG + \
                                2*(hhm - deltahm)*Lcm*(Lhm*nNTEG/shaghm) + \
                                2*(hhm - deltahm)*deltahm*(Lhm*nNTEG/shaghm)*(math.trunc(Lcm/ledhm) + 1) + \
                                (shaghm - deltahm)*deltahm*(Lhm*nNTEG/shaghm)*math.trunc(Lcm/ledhm) + \
                                2*shaghm*deltahm*(Lhm*nNTEG/shaghm)
                        Arhm1 = Arrowhm/(2*nNTEG*nTEMrowhm)
                        Arthmelem = Arthm/(2*nNTEG*nTEMrowhm*rownumtot)
                        Arrowcm = (hcmm - deltacm)*(shagcm - deltacm)*(Lcm*nNTEG/shagcm)
                        Artcm = 2*Lhm*Lcm*(nNTEG - 1) - 2*deltacm*Lhm*(Lcm*(nNTEG - 1)/shagcm) + 2*hcmm*Lhm*(nNTEG - 1) + \
                            2*(hcmm - deltacm)*Lhm*(Lcm*(nNTEG - 1)/shagcm) + \
                            2*(hcmm - deltacm)*deltacm*(Lcm*(nNTEG - 1)/shagcm) + \
                            2*shagcm*deltacm*(Lcm*(nNTEG - 1)/shagcm) + 2*Lhm*Lcm*2 - 2*deltacm*Lhm*(Lcm*2/shagcm) + \
                            2*hcms*Lhm*2 + 2*(hcms - deltacm)*Lhm*(Lcm*2/shagcm) + 2*(hcms - deltacm)*deltacm*(Lcm*2/shagcm) + \
                            2*shagcm*deltacm*(Lcm*2/shagcm)
                        dhidrcm = 4*Arrowcm*Lhm/Artcm
                        Arcm1 = Arrowcm/(2*nNTEG*rownumtot)
                        Artcmelem = Artcm/(2*nNTEG*nTEMrowhm*rownumtot)
                        Artosn = Lhm*Lcm/(nTEMrowhm*rownumtot)
                        Thm = Thmin
                        Tcm = Tcmin
                        whm = whmin
                        wcm = wcmin
                        if fin_type == 'Continuous':
                            Tplhm = Thm - (Thm - Tcm)/4
                            Tsthm = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                ((Thm + Tplhm)/2)*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                        Tplcm = Tcm + (Thm - Tcm)/4
                        Tstcm = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                            ((Tcm + Tplcm)/2)*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                        dTout1suc = 0
                        Thmout = Thmin
                        Tcmout = Tcmin
                        # определение температур на выходе в режиме ХХ
                        while dTout1suc == 0:
                            rohmm = rohm0*Thm0/Thm
                            nuhmm = nuex(Thm, rH2O)
                            lbdhmm = lbdex(Thm, rH2O)
                            chmm = cpex(Thm, rH2O, lbdhmm, rohmm, nuhmm)
                            Prhmm = nuhmm*rohmm*chmm/lbdhmm
                            Reyhm = whm*dhidrhm/nuhmm

                            # сплошное оребрение
                            if fin_type == 'Continuous':
                                if Reyhm <= 2300:
                                    Tstsuc = 0
                                    # уточнение осреднённой температуры металла между оребрением и основанием
                                    while Tstsuc == 0:
                                        rohmst = rohm0*Thm0/Tsthm
                                        nuhmst = nuex(Tsthm, rH2O)
                                        alphateplhm = 1.86 * (Reyhm*Prhmm/(Lcm/dhidrhm))**(1/3) * (rohmm*nuhmm/(rohmst*nuhmst))**0.14 * lbdhmm / dhidrhm
                                        mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5 / deltahm
                                        Tfinhm = Thm - (Thm - Tplhm)*math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                        Tsthm1 = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                            Tfinhm*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                                        delTst = abs(Tsthm - Tsthm1)
                                        delTst1 = Tsthm - Tsthm1
                                        if delTst > 0.1:
                                            Tsthm = Tsthm - 0.05 if delTst1 > 0 else Tsthm + 0.05
                                        else:
                                            Tstsuc = 1
                                else:
                                    fric = 1/(1.58*math.log(Reyhm) - 3.28)**2
                                    alphateplhm = ((fric/2)*(Reyhm - 1000)*Prhmm/(1 + 12.7*(fric/2)**0.5*(Prhmm**(2/3) - 1)))*lbdhmm/dhidrhm
                                    mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5/deltahm
                                etafhm = math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                etafhmtot = 1 - Arfhm*(1 - etafhm)/Arthm
                            # рассечное оребрение
                            else:
                                if Reyhm <= ReyNus:
                                    Nushm = 4.37*0.0001*(deltahm/dhidrhm)**(-2.6)*(ledhm/dhidrhm)**(-0.15)*Reyhm**(2.2*(deltahm/dhidrhm)**0.55*(ledhm/dhidrhm)**(-0.02))
                                else:
                                    Nushm = 7.23*0.001*(deltahm/dhidrhm)**(-1.6)*(ledhm/dhidrhm)**(-0.9)*Reyhm**(1.2*(deltahm/dhidrhm)**0.34*(ledhm/dhidrhm)**0.15)
                                alphateplhm = Nushm*lbdhmm/dhidrhm
                                mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5/deltahm
                            alphateplfhm = alphateplhm*etafhmtot

                            rocmm = row(Tcm)
                            nucmm = nuw(Tcm, rocmm)
                            lbdcmm = lbdw(Tcm)
                            ccmm = cpw(Tcm)
                            Prcmm = ccmm*rocmm*nucmm/lbdcmm
                            Reycm = wcm*dhidrcm/nucmm
                            if Reycm <= 2300:
                                Tstsuc = 0
                                # уточнение осреднённой температуры металла между оребрением и основанием
                                while Tstsuc == 0:
                                    rocmst = row(Tstcm)
                                    nucmst = nuw(Tstcm, rocmm)
                                    alphateplcm = 1.86 * (Reycm*Prcmm/(Lhm/dhidrcm))**(1/3) * (rocmm*nucmm/(rocmst*nucmst))**0.14 * lbdcmm/dhidrcm
                                    mfcm = (2*alphateplcm*deltacm/lbdOTEG)**0.5 / deltacm
                                    Tfincm = Tcm + (Tplcm - Tcm)*math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                                    Tstcm1 = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                                        Tfincm*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                                    delTst = abs(Tstcm - Tstcm1)
                                    delTst1 = Tstcm - Tstcm1
                                    if delTst > 0.1:
                                        Tstcm = Tstcm - 0.05 if delTst1 > 0 else Tstcm + 0.05
                                    else:
                                        Tstsuc = 1
                            else:
                                fric = 1/(1.58*math.log(Reycm) - 3.28)**2
                                alphateplcm = ((fric/2)*(Reycm - 1000)*Prcmm/(1 + 12.7*(fric/2)**0.5*(Prcmm**(2/3) - 1))) * lbdcmm/dhidrcm
                                mfcm = (2*alphateplcm*deltacm/lbdOTEG)**0.5/deltacm
                            etafcm = math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                            etafcmtot = 1 - Arfcm*(1 - etafcm)/Artcm
                            alphateplfcm = alphateplcm*etafcmtot

                            UA = 1/(
                                (1/(kkonvhm*alphateplfhm*Arthmelem)) + (bsthichm/(lbdNTEG*Artosn)) + (hTEM/(lbd0*nTE*ArTE)) + \
                                2*(0.0001/(ktp*0.8*Arcer)) + (bsthiccm/(lbdOTEG*Artosn)) + (1/(kkonvcm*alphateplfcm*Artcmelem))
                            )
                            temkGhm = Arhm1*whm*rohmm*chmm
                            temkGcm = Arcm1*wcm*rocmm*ccmm
                            if temkGhm < temkGcm:
                                temkGmin = temkGhm
                                Cr = temkGhm/temkGcm
                            else:
                                temkGmin = temkGcm
                                Cr = temkGcm/temkGhm
                            NTU = UA/temkGmin
                            # обе среды не перемешиваются
                            if Reyhm < 2300:
                                eps = 1 - math.exp(NTU**0.22*(math.exp(-1*Cr*NTU**0.78) - 1)/Cr)
                            else: # перемешивается горячая среда
                                if temkGhm < temkGcm:
                                    eps = 1 - math.exp((-1/Cr)*(1 - math.exp(-1*Cr*NTU)))
                                else:
                                    eps = (1/Cr)*(1 - math.exp(-1*Cr*(1 - math.exp(-1*NTU))))
                            Qhm = eps*temkGmin*(Thmin - Tcmin)*kiz
                            Thmout1 = Thmin - eps*(temkGmin/temkGhm)*(Thmin - Tcmin)
                            Tcmout1 = Tcmin + eps*(temkGmin/temkGcm)*(Thmin - Tcmin)*kiz
                            delThmout = abs(Thmout - Thmout1)
                            delTcmout = abs(Tcmout - Tcmout1)
                            if delThmout > 0.1 or delTcmout > 0.1:
                                if delThmout > 0.1:
                                    delThmout1 = Thmout - Thmout1
                                    Thmout = Thmout - 0.01 if delThmout1 > 0 else Thmout + 0.01
                                    Thm = (Thmin + Thmout)/2
                                if delTcmout > 0.1:
                                    delTcmout1 = Tcmout - Tcmout1
                                    Tcmout = Tcmout - 0.01 if delTcmout1 > 0 else Tcmout + 0.01
                                    Tcm = (Tcmin + Tcmout)/2
                                whmout = Vhm*Thmout/(Thm0*Arrowhm)
                                whm = (whmin + whmout)/2
                                rocmout = row(Tcmout)
                                wcmout = wcm0*rocm0/rocmout
                                wcm = (wcmin + wcmout)/2
                                if fin_type == 'Continuous' and Reyhm <= 2300:
                                    Tplhm = Thm - Qhm/(alphateplfhm*Arthmelem)
                                    Tfinhm = Thm - (Thm - Tplhm)*math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                    Tsthm = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                        Tfinhm*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                                if Reycm <= 2300:
                                    Tplcm = Tcm + Qhm/(alphateplfcm*Artcmelem)
                                    Tfincm = Tcm + (Tplcm - Tcm)*math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                                    Tstcm = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                                        Tfincm*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                            else:
                                dTout1suc = 1

                        Rhm = nTE*ArTE/(kkonvhm*alphateplfhm*Arthmelem) + bsthichm*nTE*ArTE/(lbdNTEG*Artosn) + 0.0001*nTE*ArTE/(ktp*0.8*Arcer)
                        Rcm = nTE*ArTE/(kkonvcm*alphateplfcm*Artcmelem) + bsthiccm*nTE*ArTE/(lbdOTEG*Artosn) + 0.0001*nTE*ArTE/(ktp*0.8*Arcer)
                        Tchm = Thm - Qhm*Rhm/(nTE*ArTE)
                        Tccm = Tcm + Qhm*Rcm/(nTE*ArTE)
                        kT =1 + Z0*((Tchm+Tccm)/2 + m*Tchm)/(1 + m)**2
                        dTout2suc = 0
                        # итерационные уточнения генераторного режима
                        while dTout2suc == 0:
                            deltaTc = Tchm - Tccm
                            eta = Z0*deltaTc*m/(kT*(1 + m)**2)
                            qhmplot = deltaTc*kT/R
                            Qhm = qhmplot*nTE*ArTE
                            Pelem = Qhm*eta
                            Thmout = Thmin - Qhm/(kiz*temkGhm)
                            Thm = (Thmin + Thmout)/2
                            Tchm1 = Thm - qhmplot*Rhm
                            Qcm = Qhm*(1 - eta)
                            Tcmout = Tcmin + Qcm/temkGcm
                            Tcm = (Tcmin + Tcmout)/2
                            Tccm1 = Tcm + qhmplot*(1 - eta)*Rcm
                            delTchm = abs(Tchm - Tchm1)
                            delTccm = abs(Tccm - Tccm1)
                            if delTchm > 0.1 or delTccm > 0.1:
                                if delTchm > 0.1:
                                    delTchm1 = Tchm - Tchm1
                                    Tchm = Tchm - 0.01 if delTchm1 > 0 else Tchm + 0.01
                                if delTccm > 0.1:
                                    delTccm1 = Tccm - Tccm1
                                    Tccm = Tccm - 0.01 if delTccm1 > 0 else Tccm + 0.01
                                kT = 1 + Z0*((Tchm + Tccm)/2 + m*Tchm)/(1 + m)**2
                            else:
                                dTout2suc = 1

                        whmout = Vhm*Thmout/(Thm0*Arrowhm)
                        whm = (whmin + whmout)/2
                        rocmout = row(Tcmout)
                        wcmout = wcm0*rocm0/rocmout
                        wcm = (wcmin + wcmout)/2

                        mwhmout[imas], mThmin[imas], mThmout[imas], mTcmout[imas] = whmout, Thmin, Thmout, Tcmout
                        PTEG += Pelem
                        TEMnum -= 1
                        imas += 1
                        # переход к следующему ряду
                        if TEMnum == 0:
                            delThm = (mThmin[imas - nTEMrowhm] - mThmout[imas - nTEMrowhm])*delrownumtot
                            Thmrest = mThmout[imas - nTEMrowhm] - Thmpr
                            # изменение числа рядов, либо завершение итераций
                            if Thmrest < delThm or (Thmrest >= delThm and rownum == rownumtot):
                                if (Thmrest < delThm and rownum == rownumtot) or rownum == rowmax1:
                                    teplsuc = 1
                                else: # уточнение числа рядов
                                    if Thmrest < delThm:
                                        rownumtot -= 1 # уменьшение числа рядов
                                        rowf = 1
                                    else:
                                        if rowf == 1:
                                            teplsuc = 1
                                        else:
                                            rownumtot += 1 # увеличение числа рядов
                                    # выход из расчёта по числу рядов
                                    if rownumtot < 4:
                                        teplsuc = 1
                                        calccontin = 0
                                    # подготовка к следующей итерации
                                    if calccontin == 1 and teplsuc == 0:
                                        imas = 1
                                        rownum = 1
                                        TEMnum = nTEMrowhm
                                        Thmin, Tcmin, whmin, wcmin, PTEG = Thm0, Tcm0, whm0, wcm0, 0
                                        mThmin = np.zeros(10000)
                                        mThmout = np.zeros(10000)
                                        mTcmout = np.zeros(10000)
                                        mwhmout = np.zeros(10000)
                            else: # следующий ряд
                                rownum += 1
                                TEMnum = nTEMrowhm
                        # переход к учатску, расположенному левее
                        if teplsuc == 0:
                            if TEMnum == nTEMrowhm: # параметры воды для первого участка
                                Tcmin, wcmin = Tcm0, wcm0
                            else: # параметры воды для участков левее
                                Tcmin, wcmin = Tcmout, wcmout
                            if rownum == 1: # параметры горячей среды для первого ряда
                                whmin, Thmin = whm0, Thm0
                            else: #  параметры горячей среды в последующих рядах
                                if TEMnum == nTEMrowhm:
                                    whmin = (
                                        mwhmout[imas - nTEMrowhm] + mwhmout[imas - nTEMrowhm + 1] + mwhmout[imas - nTEMrowhm + 2]
                                    )/3
                                    Thmin = (
                                        mThmout[imas - nTEMrowhm] + mThmout[imas - nTEMrowhm + 1] + mThmout[imas - nTEMrowhm + 2]
                                    )/3
                                if TEMnum < nTEMrowhm and TEMnum > 1:
                                    whmin = (
                                        mwhmout[imas - nTEMrowhm - 1] + mwhmout[imas - nTEMrowhm] + mwhmout[imas - nTEMrowhm + 1]
                                    )/3
                                    Thmin = (
                                        mThmout[imas - nTEMrowhm - 1] + mThmout[imas - nTEMrowhm] + mThmout[imas - nTEMrowhm + 1]
                                    )/3
                                if TEMnum == 1:
                                    whmin = (
                                        mwhmout[imas - nTEMrowhm - 2] + mwhmout[imas - nTEMrowhm - 1] + mwhmout[imas - nTEMrowhm]
                                    )/3
                                    Thmin = (
                                        mThmout[imas - nTEMrowhm - 2] + mThmout[imas - nTEMrowhm - 1] + mThmout[imas - nTEMrowhm]
                                    )/3
                if calccontin == 0:
                    rtcsuc = 1
                # превышение количества ТЭМ не произошло, либо произошло на скорости меньше предельной
                else:
                    # превышение количества ТЭМ не произошло
                    if nextdrow == 0:
                        widthcm = aTEM*rownumtot + deltaTEM*(rownumtot - 1) + 2*flbdist
                        ThmTEGout3, whmTEGout, im = 0, 0, 0
                        while im < nTEMrowhm:
                            ThmTEGout3 += mThmout[nTEMrowhm*rownumtot - im]
                            whmTEGout += mwhmout[nTEMrowhm*rownumtot - im]
                            im += 1
                        ThmTEGout3 /= nTEMrowhm
                        whmTEGout /= nTEMrowhm
                        rohmout = rohm0*Thm0/ThmTEGout3
                        rtcconfper = (1.183*dF3difhm**2 - 2.3038*dF3difhm + 1.1259)*whmTEGout**2*rohmout/2
                        wbefobtout = whmTEGout*dF3difhm
                        rtcconfobt = (1.183*dF2difhm**2 - 2.3038*dF2difhm + 1.1259)*0.65*wbefobtout**2*rohmout/2
                        wbefconf = wbefobtout*dF2difhm
                        rtcpatrperout = (
                            1 / (2*math.log(dpatrper/0.0004, 10) + 1.14)**2
                        ) * (dpatrhm/(2*dpatrper)) * wbefconf**2 * rohmout / 2
                        wconfout = wbefconf/dF1difhm
                        rtcconf = 0.1*wconfout**2*rohmout/2
                        rtcpatrout = (
                            1 / (2*math.log(dpatrper/0.0004, 10) + 1.14)**2
                        ) * wconfout**2 * rohmout / 4
                        ThmTEG = (Thm0 + ThmTEGout3)/2
                        rohmTEG = rohm0*Thm0/ThmTEG
                        nuhmTEG = nuex(ThmTEG, rH2O)
                        whmTEG = (whm0 + whmTEGout)/2
                        ReyhmTEG = whmTEG*dhidrhm/nuhmTEG
                        if fin_type == 'Continuous':
                            if ReyhmTEG <= 2300:
                                Gfric = Friclam(ReyhmTEG, hhm, deltahm, shaghm)
                            else:
                                Gfric = 0.196 * ReyhmTEG**(-0.2)
                        else:
                            ReyGfric = 448 * (deltahm/dhidrhm)**(-0.653) * (ledhm/dhidrhm)**0.09
                            if ReyhmTEG <= ReyGfric:
                                Gfric = 1.05 * (deltahm/dhidrhm)**(-1.05) * (ledhm/dhidrhm)**(-0.217) * ReyhmTEG**(
                                    -0.277 * (deltahm/dhidrhm)**(-0.285) * (ledhm/dhidrhm)**0.064
                                )
                            else:
                                Gfric = 0.131 * (deltahm/dhidrhm)**(-0.44) * (ledhm/dhidrhm)**(-0.234) * ReyhmTEG**(
                                    -0.0042 * (deltahm/dhidrhm)**(-1.25) * (ledhm / dhidrhm)**0.39
                                )
                        rtchmap = 1.15 * Gfric * (widthcm/dhidrhm) * whmTEG**2 * rohmTEG / 2
                        rtchm = rtcpatrent + rtcdif + rtcpatrperent + rtcdifobt + rtcdifper + rtchmap + rtcconfper + \
                            rtcconfobt + rtcpatrperout + rtcconf + rtcpatrout
                        rtcGrow = rtchmap/rownumtot

                    if rtchm > rtchmmax or nextdrow == 1:
                        if nextdrow == 0:
                            whm0 -= 1
                            uslwex0 = whm0*1000
                        # уменьшение числа ТЭМ вдоль потока ОГ
                        if whm0 < 10 or nextdrow == 1:
                            whm0 = 35 if wbefdif > 35 else wbefdif
                            uslwex0 = whm0*1000
                            if rownumtot > 4:
                                delrownumtot += 1
                                nextdrow = 0
                                rowmax1 -= 1
                            else:
                                rtcsuc, calccontin = 1, 0
                    else:
                        rtcsuc = 1

            # Обеспечено сопротивление по горячей среде
            if calccontin == 1:

                # проверка гидравлической схемы
                # по холодной среде
                delTcmmax = Tcmpr - Tcm0
                jdTW, delT_m, w_0 = 0, 0, wcm0
                for idTw in range(1, rownumtot + 1):
                    jdTW = jdTW + nTEMrowhm
                    delT_m = delT_m + mTcmout[jdTW] - Tcm0
                delT_m = delT_m/rownumtot  # средний перепад температур
                lfw, flownum, flownum1, cannummax = 0, 1, 1, math.trunc(nNTEG)
                # определение числа паралленых групп каналов
                while lfw == 0:
                    flownum1 += 1
                    nummax11 = nNTEG/flownum1 - math.trunc(nNTEG/flownum1)
                    if nummax11 == 0:
                        cannummax1 = math.trunc(nNTEG/flownum1)
                    else:
                        cannummax1 = math.trunc(nNTEG/flownum1) + 1
                    delT_m1 = delTcmmax/cannummax1
                    w_01 = wcment*delT_m/delT_m1
                    if w_01 < wcment:
                        lfw = 1
                    else:
                        flownum = flownum1
                        cannummax = cannummax1
                        w_0 = w_01
                        if flownum > 12: # количество групп с запасом
                            lfw = 1
                            calccontin = 0
                if calccontin == 1:
                    ro_in = row(Tcm0)
                    ro_out = row(Tcmpr)
                    w_out = w_0*ro_in/ro_out
                    w_ch = (w_0 + w_out)/2
                    ro_m = (ro_in + ro_out)/2
                    TcmTEG = (Tcm0 + Tcmpr)/2
                    nu_m = nuw(TcmTEG, ro_m)
                    Rey_ch = w_ch*dhidrcm/nu_m
                    if Rey_ch <= 2300:
                        Gfric = Friclam(Rey_ch, hcmm, deltacm, shagcm)
                    else:
                        Gfric = 0.196 * Rey_ch**(-0.2)
                    rtcwsuc = 0
                    # гидравлический расчёт и уточнение числа параллельных групп каналов
                    while rtcwsuc == 0:
                        rtc_ap = Gfric * (widthhm / dhidrcm) * w_ch**2 * ro_m / 2
                        rtc_ch = 1.15*(
                            cannummax*rtc_ap + (cannummax - 1)*0.48*ro_m*2 + cannummax*(
                                1.1*ro_m*2 + 0.5 * ro_m * w_ch**2 * 0.5 + 1 * ro_m * w_ch**2 * 0.5 + 0.5*ro_m*2
                            )
                        )
                        if rtc_ch <= rtcwmax:
                            rtcwsuc = 1
                        else:
                            flownum += 1
                            nummax11 = nNTEG/flownum - math.trunc(nNTEG/flownum)
                            if nummax11 == 0:
                                cannummax = math.trunc(nNTEG/flownum)
                            else:
                                cannummax = math.trunc(nNTEG/flownum) + 1
                            w_0 = wcment
                            w_out = w_0*ro_in/ro_out
                            w_ch = (w_0 + w_out)/2
                            Rey_ch = w_ch*dhidrcm/nu_m
                            if Rey_ch <= 2300:
                                Gfric = Friclam(Rey_ch, hcmm, deltacm, shagcm)
                            else:
                                Gfric = 0.196 * Rey_ch**(-0.2)
                        if flownum > 12: # количество групп с запасом
                            rtcwsuc = 1
                    if flownum > 12: # количество групп с запасом
                        calccontin = 0

                # проверка показала возможность реализации
                # гидравлической схемы по ХС
                if calccontin == 1:
                    wcmin, whmin, Tcmin, Thmin = wcm0, whm0, Tcm0, Thm0
                    teplsuc = 0
                    imas, imas1, rownum, rownumtot1, rowf, PTEG1 = 1, 1, 1, rownumtot, 0, 0
                    mThmin = np.zeros(10000)
                    mThmout = np.zeros(10000)
                    mTcmout = np.zeros(10000)
                    mwhmout = np.zeros(10000)
                    mTEMisp, nTEMrowrezm = layer_distr_func(
                        distr_perc=tem_distr[0], rownumtot=rownumtot1,
                        nTEMrow=nTEMrowhm, Lpolez=Lpolez
                    )
                    TEMnum, TEMnum1 = nTEMrowrezm[1], nTEMrowrezm[1]

                    # тепловой расчёт рядами по ОГ
                    while teplsuc == 0:
                        Lcm = aTEM*rownumtot1 + deltaTEM*(rownumtot1 - 1) + 2*deltaedg
                        Arfhm = nNTEG*math.trunc(Lhm/shaghm)*2*(hhm - deltahm)*Lcm
                        Arfcm = math.trunc(Lcm/shagcm)*2*Lhm*((nNTEG - 1)*(hcmm - deltacm) + 2*(hcms - deltacm))

                        TEMqua, TEMpr1, TEMpr2 = 0, 0, 0
                        if imas1 > 1:
                            imas2 = imas1 - 1
                        # число пустых слотов вокруг расчётного ТЭМ
                        for jm in range(1, nTEMrowhm+1):
                            masparam = mTEMisp[rownum, jm]
                            if masparam == 1:
                                TEMqua += 1
                            if imas1 == 1: # для первого ТЭМ в ряду
                                if masparam == 0  and TEMqua == imas1:
                                    TEMpr2 += 1
                            if imas1 == TEMnum1: # для последнего ТЭМ в ряду
                                if masparam == 0 and TEMqua == imas2:
                                    TEMpr1 += 1
                            # для промежуточного ТЭМ
                            if imas1 > 1 and imas1 < TEMnum1:
                                if masparam == 0 and TEMqua == imas2:
                                    TEMpr1 += 1
                                if masparam == 0  and TEMqua == imas1:
                                    TEMpr2 += 1
                        if imas1 == 1:
                            nslot = 1 + TEMpr2/2
                        if imas1 == TEMnum1:
                            nslot = 1 + TEMpr1/2
                        if imas1 > 1 and imas1 < TEMnum1:
                            nslot = 1 + TEMpr1/2 + TEMpr2/2

                        if fin_type == 'Continuous':
                            Arthm = 2*Lhm*Lcm*nNTEG - 2*deltahm*Lcm*(Lhm*nNTEG/shaghm) + \
                                2*hhm*Lcm*nNTEG + 2*(hhm - deltahm)*Lcm*(Lhm*nNTEG/shaghm) + \
                                2*(hhm - deltahm)*deltahm*(Lhm*nNTEG/shaghm) + 2*shaghm*deltahm*(Lhm*nNTEG/shaghm)
                            dhidrhm = 4*Arrowhm*Lcm/Arthm
                        else:
                            Arthm = 2*Lhm*Lcm*nNTEG - 2*deltahm*Lcm*(Lhm*nNTEG/shaghm) + 2*hhm*Lcm*nNTEG + \
                                2*(hhm - deltahm)*Lcm*(Lhm*nNTEG/shaghm) + \
                                2*(hhm - deltahm)*deltahm*(Lhm*nNTEG/shaghm)*(math.trunc(Lcm/ledhm) + 1) + \
                                (shaghm - deltahm)*deltahm*(Lhm*nNTEG/shaghm)*math.trunc(Lcm/ledhm) + \
                                2*shaghm*deltahm*(Lhm*nNTEG/shaghm)
                        Arhm1 = Arrowhm*nslot/(2*nNTEG*nTEMrowhm)
                        Arthmelem = Arthm*nslot/(2*nNTEG*nTEMrowhm*rownumtot1)
                        Arrowcm = (hcmm - deltacm)*(shagcm - deltacm)*(Lcm*nNTEG/shagcm)
                        Artcm = 2*Lhm*Lcm*(nNTEG - 1) - 2*deltacm*Lhm*(Lcm*(nNTEG - 1)/shagcm) + 2*hcmm*Lhm*(nNTEG - 1) + \
                            2*(hcmm - deltacm)*Lhm*(Lcm*(nNTEG - 1)/shagcm) + \
                            2*(hcmm - deltacm)*deltacm*(Lcm*(nNTEG - 1)/shagcm) + \
                            2*shagcm*deltacm*(Lcm*(nNTEG - 1)/shagcm) + 2*Lhm*Lcm*2 - 2*deltacm*Lhm*(Lcm*2/shagcm) + \
                            2*hcms*Lhm*2 + 2*(hcms - deltacm)*Lhm*(Lcm*2/shagcm) + 2*(hcms - deltacm)*deltacm*(Lcm*2/shagcm) + \
                            2*shagcm*deltacm*(Lcm*2/shagcm)
                        dhidrcm = 4*Arrowcm*Lhm/Artcm
                        Arcm1 = Arrowcm/(2*nNTEG*rownumtot1)
                        Artcmelem = Artcm*nslot/(2*nNTEG*nTEMrowhm*rownumtot1)
                        Artosn = Lhm*Lcm*nslot/(nTEMrowhm*rownumtot1)
                        Thm = Thmin
                        Tcm = Tcmin
                        whm = whmin
                        wcm = wcmin
                        if fin_type == 'Continuous':
                            Tplhm = Thm - (Thm - Tcm)/4
                            Tsthm = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                ((Thm + Tplhm)/2)*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                        Tplcm = Tcm + (Thm - Tcm)/4
                        Tstcm = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                            ((Tcm + Tplcm)/2)*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                        dTout1suc = 0
                        Thmout = Thmin
                        Tcmout = Tcmin
                        # определение температур на выходе в режиме ХХ
                        while dTout1suc == 0:
                            rohmm = rohm0*Thm0/Thm
                            nuhmm = nuex(Thm, rH2O)
                            lbdhmm = lbdex(Thm, rH2O)
                            chmm = cpex(Thm, rH2O, lbdhmm, rohmm, nuhmm)
                            Prhmm = nuhmm*rohmm*chmm/lbdhmm
                            Reyhm = whm*dhidrhm/nuhmm

                            # сплошное оребрение
                            if fin_type == 'Continuous':
                                if Reyhm <= 2300:
                                    Tstsuc = 0
                                    # уточнение осреднённой температуры металла между оребрением и основанием
                                    while Tstsuc == 0:
                                        rohmst = rohm0*Thm0/Tsthm
                                        nuhmst = nuex(Tsthm, rH2O)
                                        alphateplhm = 1.86 * (Reyhm*Prhmm/(Lcm/dhidrhm))**(1/3) * (rohmm*nuhmm/(rohmst*nuhmst))**0.14 * lbdhmm / dhidrhm
                                        mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5 / deltahm
                                        Tfinhm = Thm - (Thm - Tplhm)*math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                        Tsthm1 = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                            Tfinhm*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                                        delTst = abs(Tsthm - Tsthm1)
                                        delTst1 = Tsthm - Tsthm1
                                        if delTst > 0.1:
                                            Tsthm = Tsthm - 0.05 if delTst1 > 0 else Tsthm + 0.05
                                        else:
                                            Tstsuc = 1
                                else:
                                    fric = 1 / (1.58*math.log(Reyhm) - 3.28)**2
                                    alphateplhm = ((fric/2)*(Reyhm - 1000)*Prhmm/(1 + 12.7*(fric/2)**0.5*(Prhmm**(2/3) - 1)))*lbdhmm/dhidrhm
                                    mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5/deltahm
                                etafhm = math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                etafhmtot = 1 - Arfhm*(1 - etafhm)/Arthm
                            else:
                                if Reyhm <= ReyNus:
                                    Nushm = 4.37*0.0001*(deltahm/dhidrhm)**(-2.6)*(ledhm/dhidrhm)**(-0.15)*Reyhm**(2.2*(deltahm/dhidrhm)**0.55*(ledhm/dhidrhm)**(-0.02))
                                else:
                                    Nushm = 7.23*0.001*(deltahm/dhidrhm)**(-1.6)*(ledhm/dhidrhm)**(-0.9)*Reyhm**(1.2*(deltahm/dhidrhm)**0.34*(ledhm/dhidrhm)**0.15)
                                alphateplhm = Nushm*lbdhmm/dhidrhm
                                mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5/deltahm
                            alphateplfhm = alphateplhm*etafhmtot

                            rocmm = row(Tcm)
                            nucmm = nuw(Tcm, rocmm)
                            lbdcmm = lbdw(Tcm)
                            ccmm = cpw(Tcm)
                            Prcmm = ccmm * rocmm * nucmm / lbdcmm
                            Reycm = wcm * dhidrcm / nucmm
                            if Reycm <= 2300:
                                Tstsuc = 0
                                # уточнение осреднённой температуры металла между оребрением и основанием
                                while Tstsuc == 0:
                                    rocmst = row(Tstcm)
                                    nucmst = nuw(Tstcm, rocmm)
                                    alphateplcm = 1.86 * (Reycm*Prcmm/(Lhm/dhidrcm))**(1/3) * (rocmm*nucmm/(rocmst*nucmst))**0.14 * lbdcmm/dhidrcm
                                    mfcm = (2*alphateplcm*deltacm/lbdOTEG)**0.5 / deltacm
                                    Tfincm = Tcm + (Tplcm - Tcm)*math.tanh(mfcm *((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                                    Tstcm1 = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                                        Tfincm*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                                    delTst = abs(Tstcm - Tstcm1)
                                    delTst1 = Tstcm - Tstcm1
                                    if delTst > 0.1:
                                        Tstcm = Tstcm - 0.05 if delTst1 > 0 else Tstcm + 0.05
                                    else:
                                        Tstsuc = 1
                            else:
                                fric = 1/(1.58*math.log(Reycm) - 3.28)**2
                                alphateplcm = ((fric/2)*(Reycm - 1000)*Prcmm/(1 + 12.7*(fric/2)**0.5*(Prcmm**(2/3) - 1))) * lbdcmm/dhidrcm
                                mfcm = (2*alphateplcm*deltacm/lbdOTEG)**0.5/deltacm
                            etafcm = math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                            etafcmtot = 1 - Arfcm*(1 - etafcm)/Artcm
                            alphateplfcm = alphateplcm*etafcmtot
                            UA = 1/(
                                (1/(kkonvhm*alphateplfhm*Arthmelem)) + (bsthichm/(lbdNTEG*Artosn)) + (hTEM/(lbd0*nTE*ArTE)) + \
                                2*(0.0001/(ktp*0.8*Arcer)) + (bsthiccm/(lbdOTEG*Artosn)) + (1/(kkonvcm*alphateplfcm*Artcmelem))
                            )
                            temkGhm = Arhm1*whm*rohmm*chmm
                            temkGcm = Arcm1*wcm*rocmm*ccmm
                            if temkGhm < temkGcm:
                                temkGmin = temkGhm
                                Cr = temkGhm/temkGcm
                            else:
                                temkGmin = temkGcm
                                Cr = temkGcm/temkGhm
                            NTU = UA/temkGmin
                            # обе среды не перемешиваются
                            if Reyhm < 2300:
                                eps = 1 - math.exp(NTU**0.22*(math.exp(-1*Cr*NTU**0.78) - 1)/Cr)
                            else:  # перемешивается горячая среда
                                if temkGhm < temkGcm:
                                    eps = 1 - math.exp((-1/Cr)*(1 - math.exp(-1*Cr*NTU)))
                                else:
                                    eps = (1/Cr)*(1 - math.exp(-1*Cr*(1 - math.exp(-1*NTU))))
                            Qhm = eps*temkGmin*(Thmin - Tcmin)*kiz
                            Thmout1 = Thmin - eps*(temkGmin/temkGhm)*(Thmin - Tcmin)
                            Tcmout1 = Tcmin + eps*(temkGmin/temkGcm)*(Thmin - Tcmin)*kiz
                            delThmout = abs(Thmout - Thmout1)
                            delTcmout = abs(Tcmout - Tcmout1)
                            if delThmout > 0.1 or delTcmout > 0.1:
                                if delThmout > 0.1:
                                    delThmout1 = Thmout - Thmout1
                                    Thmout = Thmout - 0.01 if delThmout1 > 0 else Thmout + 0.01
                                    Thm = (Thmin + Thmout)/2
                                if delTcmout > 0.1:
                                    delTcmout1 = Tcmout - Tcmout1
                                    Tcmout = Tcmout - 0.01 if delTcmout1 > 0 else Tcmout + 0.01
                                    Tcm = (Tcmin + Tcmout)/2
                                whmout = Vhm*Thmout/(Thm0 * Arrowhm)
                                whm = (whmin + whmout)/2
                                rocmout = row(Tcmout)
                                wcmout = wcm0*rocm0/rocmout
                                wcm = (wcmin + wcmout)/2
                                if fin_type == 'Continuous' and Reyhm <= 2300:
                                    Tplhm = Thm - Qhm/(alphateplfhm*Arthmelem)
                                    Tfinhm = Thm - (Thm - Tplhm)*math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                    Tsthm = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                        Tfinhm*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                                if Reycm <= 2300:
                                    Tplcm = Tcm + Qhm/(alphateplfcm*Artcmelem)
                                    Tfincm = Tcm + (Tplcm - Tcm)*math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                                    Tstcm = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                                        Tfincm*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                            else:
                                dTout1suc = 1

                        Rhm = nTE*ArTE/(kkonvhm*alphateplfhm*Arthmelem) + bsthichm*nTE*ArTE/(lbdNTEG*Artosn) + 0.0001*nTE*ArTE/(ktp*0.8*Arcer)
                        Rcm = nTE*ArTE/(kkonvcm*alphateplfcm*Artcmelem) + bsthiccm*nTE*ArTE/(lbdOTEG*Artosn) + 0.0001*nTE*ArTE/(ktp*0.8*Arcer)
                        Tchm = Thm - Qhm*Rhm/(nTE*ArTE)
                        Tccm = Tcm + Qhm*Rcm/(nTE*ArTE)
                        kT = 1 + Z0*((Tchm + Tccm)/2 + m*Tchm)/(1 + m)**2
                        dTout2suc = 0
                        # итерационные уточнения генераторного режима
                        while dTout2suc == 0:
                            deltaTc = Tchm - Tccm
                            eta = Z0*deltaTc*m/(kT*(1 + m)**2)
                            qhmplot = deltaTc*kT/R
                            Qhm = qhmplot*nTE*ArTE
                            Pelem = Qhm*eta
                            Thmout = Thmin - Qhm/(kiz*temkGhm)
                            Thm = (Thmin + Thmout)/2
                            Tchm1 = Thm - qhmplot*Rhm
                            Qcm = Qhm*(1 - eta)
                            Tcmout = Tcmin + Qcm/temkGcm
                            Tcm = (Tcmin + Tcmout)/2
                            Tccm1 = Tcm + qhmplot*(1 - eta)*Rcm
                            delTchm = abs(Tchm - Tchm1)
                            delTccm = abs(Tccm - Tccm1)
                            if delTchm > 0.1 or delTccm > 0.1:
                                if delTchm > 0.1:
                                    delTchm1 = Tchm - Tchm1
                                    Tchm = Tchm - 0.01 if delTchm1 > 0 else Tchm + 0.01
                                if delTccm > 0.1:
                                    delTccm1 = Tccm - Tccm1
                                    Tccm = Tccm - 0.01 if delTccm1 > 0 else Tccm + 0.01
                                kT = 1 + Z0*((Tchm + Tccm)/2 + m*Tchm)/(1 + m)**2
                            else:
                                dTout2suc = 1

                        whmout = Vhm*Thmout/(Thm0*Arrowhm)
                        whm = (whmin + whmout)/2
                        rocmout = row(Tcmout)
                        wcmout = wcm0*rocm0/rocmout
                        wcm = (wcmin + wcmout)/2

                        mwhmout[imas], mThmin[imas], mThmout[imas], mTcmout[imas] = whmout, Thmin, Thmout, Tcmout
                        PTEG1 += Pelem
                        TEMnum -= 1
                        imas += 1
                        imas1 += 1

                        # переход к следующему ряду
                        if TEMnum == 0:
                            Thmrest = mThmout[imas - TEMnum1] - Thmpr
                            delThm = mThmin[imas - TEMnum1] - mThmout[imas - TEMnum1]
                            rtcGsv = rtcpatrent + rtcdif + rtcpatrperent + rtcdifobt + rtcdifper + rtcconfper + rtcconfobt + rtcpatrperout + rtcconf + rtcpatrout + rtcGrow*rownum
                            rtcGrest = rtchmmax - rtcGsv
                            # изменение числа рядов, либо завершение итераций
                            if (Thmrest < delThm or rtcGrest < rtcGrow) or (
                                Thmrest >= delThm and rtcGrest >= rtcGrow and rownum == rownumtot1
                            ):
                                # нет запаса по температуре или сопротивлению
                                if Thmrest < delThm or rtcGrest < rtcGrow:
                                    if rownum == rownumtot1: # завершение итераций теплового расчёта
                                        teplsuc = 1
                                    else: # уточнение числа рядов
                                        rownumtot1 -= 1
                                        rowf = 1
                                # есть запас по температуре и сопротивлению
                                if Thmrest >= delThm and rtcGrest >= rtcGrow:
                                    if rownum == rowmax: # превышено число рядов
                                        teplsuc, calccontin = 1, 0
                                    else: # уточнение числа рядов
                                        if rowf == 1:
                                            teplsuc = 1
                                        else:
                                            rownumtot1 += 1
                                            mTEMisp, nTEMrowrezm = layer_distr_func(
                                                distr_perc=tem_distr[0], rownumtot=rownumtot1,
                                                nTEMrow=nTEMrowhm, Lpolez=Lpolez
                                            )

                                # подготовка к следующей итерации
                                if teplsuc == 0:
                                    imas, imas1, rownum = 1, 1, 1
                                    TEMnum, TEMnum1 = nTEMrowrezm[1], nTEMrowrezm[1]
                                    Thmin, Tcmin, whmin, wcmin, PTEG1 = Thm0, Tcm0, whm0, wcm0, 0
                                    mThmin = np.zeros(10000)
                                    mThmout = np.zeros(10000)
                                    mTcmout = np.zeros(10000)
                                    mwhmout = np.zeros(10000)
                            else: # следующий ряд
                                rownum += 1
                                TEMnum, TEMnum1 = nTEMrowrezm[rownum], nTEMrowrezm[rownum]
                                imas1 = 1

                        # переход к учатску, расположенному левее
                        if teplsuc == 0:
                            # параметры воды для первого участка
                            if TEMnum == TEMnum1:
                                Tcmin = Tcm0
                                wcmin = wcm0
                            # параметры воды для участков левее
                            else:
                                Tcmin = Tcmout
                                wcmin = wcmout
                            # параметры газа для первого ряда
                            if rownum == 1:
                                whmin = whm0
                                Thmin = Thm0
                            # параметры газа в последующих рядах
                            else:
                                TEMqua, slot1, slot2, chon, slotnum = 0, 0, 0, 0, 0
                                # наличие пустых слотов на границе участка
                                while chon == 0:
                                    slotnum += 1
                                    masparam = mTEMisp[rownum, slotnum]
                                    if masparam == 1:
                                        TEMqua += 1
                                    if TEMqua == imas1:
                                        chon = 1
                                        if imas1 == 1:
                                            slot2 = mTEMisp[rownum, slotnum+1];
                                        if imas1 == TEMnum1:
                                            slot1 = mTEMisp[rownum, slotnum-1];
                                        if imas1 > 1 and imas1 < TEMnum1:
                                            slot1, slot2 = mTEMisp[rownum, slotnum-1], mTEMisp[rownum, slotnum+1]

                                TEMqua, TEMpr1, TEMpr2, slotnum, slotnum1, slotnum2, kisp1, kisp2, TEMslot = 0, 0, 0, 0, 0, 0, 0, 0, 0
                                # длины промежутков - TEMpr1, TEMpr2; граничные номера - slotnum1, slotnum2
                                for jm in range(1, nTEMrowhm+1):
                                    slotnum += 1
                                    masparam = mTEMisp[rownum, jm]
                                    if masparam == 1:
                                        TEMqua += 1
                                    if masparam == 1  and TEMqua == imas1:
                                        TEMslot = slotnum

                                    if imas1 == 1:
                                        slotnum1 = 1
                                        if slot2 == 0:
                                            masparam = mTEMisp[rownum, jm]
                                            if masparam == 0 and TEMqua == imas1:
                                                TEMpr2 += 1
                                                slotnum2 = slotnum
                                        else:
                                            slotnum2 = 1

                                    if imas1 == TEMnum1:
                                        slotnum2 = nTEMrowhm
                                        if slot1 == 0:
                                            imas2 = imas1 - 1
                                            masparam = mTEMisp[rownum, jm]
                                            if masparam == 0 and TEMqua == imas2:
                                                TEMpr1 += 1
                                                if TEMpr1 == 1:
                                                    slotnum1 = slotnum
                                        else:
                                            slotnum1 = nTEMrowhm

                                    if imas1 > 1 and imas1 < TEMnum1:
                                        if slot1 == 0:
                                            imas2 = imas1 - 1
                                            masparam = mTEMisp[rownum, jm]
                                            if masparam == 0 and TEMqua == imas2:
                                                TEMpr1 += 1
                                                if TEMpr1 == 1:
                                                    slotnum1 = slotnum
                                        else:
                                            slotnum1 = TEMslot
                                        if slot2 == 0:
                                            masparam = mTEMisp[rownum, jm];
                                            if masparam == 0 and TEMqua == imas1:
                                                TEMpr2 += 1
                                                slotnum2 = slotnum
                                        else:
                                            slotnum2 = TEMslot

                                # определение коэффициентов использования и крайних слотов
                                if imas1 == 1:
                                    kisp1 = 1
                                    if slot2 == 0:
                                        TEMpr2 /= 2
                                        TEMpr21 = TEMpr2 - math.trunc(TEMpr2)
                                        if TEMpr21 == 0:
                                            slotnum2 -= round(TEMpr2)
                                            kisp2 = 1
                                        else:
                                            slotnum2 -= math.trunc(TEMpr2)
                                            kisp2 = 0
                                    else:
                                        kisp2 = 1

                                if imas1 == TEMnum1:
                                    kisp2 = 1
                                    if slot1 == 0:
                                        TEMpr1 /= 2
                                        TEMpr11 = TEMpr1 - math.trunc(TEMpr1)
                                        if TEMpr11 == 0:
                                            slotnum1 = slotnum1 + round(TEMpr1)
                                            kisp1 = 1
                                        else:
                                            slotnum1 += math.trunc(TEMpr1)
                                            kisp1 = 0
                                    else:
                                        kisp1 = 1

                                if imas1 > 1 and imas1 < TEMnum1:
                                    if slot1 == 0:
                                        TEMpr1 /= 2
                                        TEMpr11 = TEMpr1 - math.trunc(TEMpr1)
                                        if TEMpr11 == 0:
                                            slotnum1 += round(TEMpr1)
                                            kisp1 = 1
                                        else:
                                            slotnum1 += math.trunc(TEMpr1)
                                            kisp1 = 0
                                    else:
                                        kisp1 = 1
                                    if slot2 == 0:
                                        TEMpr2 /= 2
                                        TEMpr21 = TEMpr2 - math.trunc(TEMpr2)
                                        if TEMpr21==0:
                                            slotnum2 -= round(TEMpr2)
                                            kisp2 = 1
                                        else:
                                            slotnum2 -= math.trunc(TEMpr2)
                                            kisp2 = 0
                                    else:
                                        kisp2 = 1

                                if kisp1 == 1 and kisp2 == 1:
                                    kispcent = 1/(slotnum2 - slotnum1 + 1)
                                    kispkr1 = kispcent
                                    kispkr2 = kispcent
                                if kisp1 == 0 and kisp2 == 0:
                                    kispcent = 1/(slotnum2 - slotnum1)
                                    kispkr1 = kispcent/2
                                    kispkr2 = kispkr1
                                if kisp1 == 0 and kisp2 == 1:
                                    kispcent = 1/(slotnum2 - slotnum1 + 0.5)
                                    kispkr1 = kispcent/2
                                    kispkr2 = kispcent
                                if kisp1 == 1 and kisp2 == 0:
                                    kispcent = 1/(slotnum2 - slotnum1 + 0.5)
                                    kispkr1 = kispcent
                                    kispkr2 = kispcent/2

                                slotnum, whmin, Thmin = slotnum1, 0, 0
                                # средние параметры
                                while slotnum <= slotnum2:
                                    if slotnum == 1:
                                        slotnumpr1 = slotnum
                                        slotnumpr2 = slotnum + 1
                                        slotnumpr3 = slotnum + 2
                                    if slotnum == nTEMrowhm:
                                        slotnumpr1 = slotnum - 2
                                        slotnumpr2 = slotnum - 1
                                        slotnumpr3 = slotnum
                                    if slotnum > 1 and slotnum < nTEMrowhm:
                                        slotnumpr1 = slotnum - 1
                                        slotnumpr2 = slotnum
                                        slotnumpr3 = slotnum + 1

                                    # для первого из трёх слотов
                                    if slotnumpr1 == 1:
                                        whmoutsl1 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + 1]
                                        Thmoutsl1 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + 1]
                                    else:
                                        TEMquapr, slotnumpr = 0, 0
                                        for jm in range(1, nTEMrowhm+1):
                                            slotnumpr += 1
                                            masparam = mTEMisp[rownum - 1, jm]
                                            if masparam == 1:
                                                TEMquapr += 1
                                                TEMslotpr = slotnumpr
                                            # в слоте находится ТЭМ
                                            if masparam == 1 and slotnumpr == slotnumpr1:
                                                whmoutsl1 = mwhmout[imas-imas1-nTEMrowrezm[rownum-1]+TEMquapr]
                                                Thmoutsl1 = mThmout[imas-imas1-nTEMrowrezm[rownum-1]+TEMquapr]
                                            # пустой слот
                                            if masparam == 0 and slotnumpr == slotnumpr1:
                                                slotquapr, TEMquapr1 = 0, 0
                                                for km in range(TEMslotpr, nTEMrowhm+1):
                                                    if TEMquapr1 < 2:
                                                        masparam = mTEMisp[rownum - 1, km]
                                                        if masparam == 0:
                                                            slotquapr += 1
                                                        else:
                                                            TEMquapr1 += 1
                                                slotquapr1 = slotquapr/2 - math.trunc(slotquapr/2)
                                                if slotquapr == 1:
                                                    slotnumch = TEMslotpr + 1
                                                else:
                                                    slotnumch = TEMslotpr + round(slotquapr/2)
                                                if (slotnumpr1 < slotnumch) or (slotnumpr1 == slotnumch and slotquapr1 == 0): # слот относится к предыдущему ТЭМ
                                                    whmoutsl1 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                                    Thmoutsl1 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                                if slotnumpr1 > slotnumch: # слот относится к следующему ТЭМ
                                                    whmoutsl1 = mwhmout[imas-imas1-nTEMrowrezm[rownum-1]+TEMquapr+1]
                                                    Thmoutsl1 = mThmout[imas-imas1-nTEMrowrezm[rownum-1]+TEMquapr+1]
                                                if slotnumpr1 == slotnumch and slotquapr1 > 0: # слот относится к обоим ТЭМ
                                                    whmoutsl1 = (
                                                        mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr] + \
                                                        mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                    )/2
                                                    Thmoutsl1 = (
                                                        mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr] + \
                                                        mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                    )/2

                                    # для второго из трёх слотов
                                    TEMquapr, slotnumpr = 0, 0
                                    for jm in range(1, nTEMrowhm+1):
                                        slotnumpr += 1
                                        masparam = mTEMisp[rownum - 1, jm]
                                        if masparam == 1:
                                            TEMquapr += 1
                                            TEMslotpr = slotnumpr
                                        # в слоте находится ТЭМ
                                        if masparam == 1 and slotnumpr == slotnumpr2:
                                            whmoutsl2 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                            Thmoutsl2 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                        # пустой слот
                                        if masparam == 0 and slotnumpr == slotnumpr2:
                                            slotquapr, TEMquapr1 = 0, 0
                                            for km in range(TEMslotpr, nTEMrowhm+1):
                                                if TEMquapr1 < 2:
                                                    masparam = mTEMisp[rownum - 1, km]
                                                    if masparam == 0:
                                                        slotquapr += 1
                                                    else:
                                                        TEMquapr1 += 1
                                            slotquapr1 = slotquapr/2 - math.trunc(slotquapr/2)
                                            if slotquapr == 1:
                                                slotnumch = TEMslotpr + 1
                                            else:
                                                slotnumch = TEMslotpr + round(slotquapr/2)
                                            if (slotnumpr2 < slotnumch) or (slotnumpr2 == slotnumch and slotquapr1 == 0): # слот относится к предыдущему ТЭМ
                                                whmoutsl2 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                                Thmoutsl2 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                            if slotnumpr2 > slotnumch: # слот относится к следующему ТЭМ
                                                whmoutsl2 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                Thmoutsl2 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                            if slotnumpr2 == slotnumch and slotquapr1 > 0: # слот относится к обоим ТЭМ
                                                whmoutsl2 = (
                                                    mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr] + \
                                                    mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                )/2
                                                Thmoutsl2 = (
                                                    mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr] + \
                                                    mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                )/2

                                    # для третьего из трёх слотов
                                    if slotnumpr3 == nTEMrowhm:
                                        whmoutsl3 = mwhmout[imas - imas1]
                                        Thmoutsl3 = mThmout[imas - imas1]
                                    else:
                                        TEMquapr, slotnumpr = 0, 0
                                        for jm in range(1, nTEMrowhm+1):
                                            slotnumpr += 1
                                            masparam = mTEMisp[rownum - 1, jm]
                                            if masparam == 1:
                                                TEMquapr += 1
                                                TEMslotpr = slotnumpr
                                            if masparam == 1 and slotnumpr == slotnumpr3: # в слоте находится ТЭМ
                                                whmoutsl3 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                                Thmoutsl3 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                            if masparam == 0 and slotnumpr == slotnumpr3: # пустой слот
                                                slotquapr, TEMquapr1 = 0, 0
                                                for km in range(TEMslotpr, nTEMrowhm+1):
                                                    if TEMquapr1 < 2:
                                                        masparam = mTEMisp[rownum - 1, km]
                                                        if masparam == 0:
                                                            slotquapr += 1
                                                        else:
                                                            TEMquapr1 += 1
                                                slotquapr1 = slotquapr/2 - math.trunc(slotquapr/2)
                                                if slotquapr == 1:
                                                    slotnumch = TEMslotpr + 1
                                                else:
                                                    slotnumch = TEMslotpr + round(slotquapr/2)
                                                if (slotnumpr3 < slotnumch) or (slotnumpr3 == slotnumch and slotquapr1 == 0): # слот относится к предыдущему ТЭМ
                                                    whmoutsl3 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                                    Thmoutsl3 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr]
                                                if slotnumpr3 > slotnumch: # слот относится к следующему ТЭМ
                                                    whmoutsl3 = mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                    Thmoutsl3 = mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                if slotnumpr3 == slotnumch and slotquapr1 > 0: # слот относится к обоим ТЭМ
                                                    whmoutsl3 = (
                                                        mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr] + \
                                                        mwhmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                    )/2
                                                    Thmoutsl3 = (
                                                        mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr] + \
                                                        mThmout[imas - imas1 - nTEMrowrezm[rownum-1] + TEMquapr + 1]
                                                    )/2

                                    if slotnum == slotnum1:
                                        whmin = kispkr1*(whmoutsl1 + whmoutsl2 + whmoutsl3)/3
                                        Thmin = kispkr1*(Thmoutsl1 + Thmoutsl2 + Thmoutsl3)/3
                                    if slotnum == slotnum2 and slotnum2 > slotnum1:
                                        whmin += kispkr2*(whmoutsl1 + whmoutsl2 + whmoutsl3)/3
                                        Thmin += kispkr2*(Thmoutsl1 + Thmoutsl2 + Thmoutsl3)/3
                                    if slotnum > slotnum1 and slotnum < slotnum2:
                                        whmin += kispcent*(whmoutsl1 + whmoutsl2 + whmoutsl3)/3
                                        Thmin += kispcent*(Thmoutsl1 + Thmoutsl2 + Thmoutsl3)/3

                                    slotnum += 1

                    # число рядов не превышено
                    if calccontin == 1:
                        rtchm = rtcGsv
                        rtchmap = rtcGrow*rownumtot1
                        widthcm = aTEM*rownumtot1 + deltaTEM*(rownumtot1 - 1) + 2*flbdist

                else:
                    results_dict['message'] = 'Расчёт невозможен. Превышено сопротивление по холодной среде.'
                    return results_dict

                # число рядов не превышено
                if calccontin == 1:
                    delTcmmax = Tcmpr - Tcm0
                    jdTW, delTcmm = 0, 0
                    for idTw in range(1, rownumtot1+1):
                        jdTW = jdTW + nTEMrowrezm[idTw]
                        delTcmm = delTcmm + mTcmout[jdTW] - Tcm0
                    delTcmm = delTcmm/rownumtot1   # средний перепад температур

                    lfw, cmflownum, cmflownum1, cmcannummax,  cmcannummin = 0, 1, 1, math.trunc(nNTEG), 0
                    #определение числа паралленых групп каналов
                    while lfw == 0:
                        cmflownum1 += 1
                        cannummax11 = nNTEG/cmflownum1 - math.trunc(nNTEG/cmflownum1)
                        if cannummax11 == 0:
                            cmcannummax1 = math.trunc(nNTEG/cmflownum1)
                            cmcannummin1 = 0
                        else:
                            cmcannummax1 = math.trunc(nNTEG/cmflownum1) + 1
                            cmcannummin1 = cmcannummax1 - 1
                        delTcmm1 = delTcmmax/cmcannummax1
                        wcm01 = wcment*delTcmm/delTcmm1
                        if wcm01 < wcment:
                            lfw = 1
                        else:
                            cmflownum = cmflownum1
                            cmcannummax = cmcannummax1
                            cmcannummin = cmcannummin1
                            wcm0 = wcm01
                            if cmflownum > 10:
                                lfw = 1
                                calccontin = 0

                    # в первом приближении,
                    # расчёт гидравлической схемы - возможен
                    if calccontin == 1:
                        dpatrcmm = (2 * wcm0 * Arrowcm / (3.14*nNTEG)) ** 0.5
                        dpatrcms = (wcm0 * Arrowcm / (3.14*nNTEG)) ** 0.5
                        ntrsoedmax = math.trunc(widthcm/0.06)
                        dtrsoed = dpatrcms
                        ntrsoed = 1
                        ntrsoed1 = 1
                        lfw = 0
                        # количество и диаметр соединительных труб
                        while lfw == 0:
                            ntrsoed1 += 1
                            dtrsoed1 = (wcm0 * Arrowcm / (3.14*ntrsoed1*nNTEG)) ** 0.5
                            if dtrsoed1 >= 0.02 and ntrsoed1 <= ntrsoedmax:
                                dtrsoed = dtrsoed1
                                ntrsoed = ntrsoed1
                            else:
                                lfw = 1

                        rocmin = row(Tcm0)
                        rocmout = row(Tcmpr)
                        wcmout = wcm0*rocmin/rocmout
                        wcm = (wcm0 + wcmout)/2
                        rocmm = (rocmin + rocmout)/2
                        TcmTEG = (Tcm0 + Tcmpr)/2
                        nucmm = nuw(TcmTEG, rocmm)
                        Reycm = wcm*dhidrcm/nucmm
                        if Reycm <= 2300:
                            Gfric = Friclam(Reycm, hcmm, deltacm, shagcm)
                        else:
                            Gfric = 0.196 * Reycm**(-0.2)
                        widthpatrobv = widthcm - 2*bflb
                        rtcwsuc = 0
                        # гидравлический расчёт и уточнение числа параллельных групп каналов
                        while rtcwsuc == 0:
                            rtccmap = Gfric * (widthhm/dhidrcm) * wcm**2 * rocmm / 2
                            rtccm = 1.15 * (
                                cmcannummax*rtccmap + (cmcannummax - 1)*0.48*rocmm*2 + cmcannummax*(
                                    1.1*rocmm*2 + 0.5 * rocmm * wcm**2 * 0.5 + 1 * rocmm * wcm**2 * 0.5 + 0.5*rocmm*2
                                )
                            )
                            if rtccm <= rtcwmax:
                                rtcwsuc = 1
                            else:
                                cmflownum += 1
                                cannummax11 = nNTEG/cmflownum - math.trunc(nNTEG/cmflownum)
                                if cannummax11 == 0:
                                    cmcannummax = math.trunc(nNTEG/cmflownum)
                                    cmcannummin = 0
                                else:
                                    cmcannummax = math.trunc(nNTEG/cmflownum) + 1
                                    cmcannummin = cmcannummax - 1
                                wcm0 = wcment
                                wcmout = wcm0*rocmin/rocmout
                                wcm = (wcm0 + wcmout)/2
                                Reycm = wcm*dhidrcm/nucmm
                                if Reycm <= 2300:
                                    Gfric = Friclam(Reycm, hcmm, deltacm, shagcm)
                                else:
                                    Gfric = 0.196 * Reycm**(-0.2)
                            if cmflownum > 10:
                                rtcwsuc = 1
                        if cmflownum > 10:
                            calccontin = 0

                    # обеспечено гидравлическое сопротивление
                    if calccontin == 1:
                        cmgrmax = 0
                        # определение числа групп каналов для воды
                        nhmcan = 0
                        while nhmcan != nNTEG:
                            cmgrmax += 1
                            cmgrmin = cmflownum - cmgrmax
                            nhmcan = cmgrmax*cmcannummax + cmgrmin*cmcannummin
                        hmgrmax = hmflownum
                        params_dict = {
                            'Reycm': [], 'alphateplcm': [], 'alphateplfcm': [], 'Prcmm': [], 'lbdcmm': [],
                            'rocmm': [], 'ccmm': [], 'nucmm': [], 'wcm': [], 'Tcm': [], 'Tcmin': [],
                            'Tcmout': [], 'Tplcm': [], 'temkGcm': [], 'etafcmtot': [], 'Reyhm': [],
                            'alphateplhm': [], 'alphateplfhm': [], 'Prhmm': [], 'lbdhmm': [], 'rohmm': [],
                            'chmm': [], 'nuhmm': [], 'whm': [], 'Thm': [], 'Thmin': [], 'Thmout': [], 'Tplhm': [],
                            'temkGhm': [], 'etafhmtot': [], 'UA': [], 'NTU': [], 'eps': [], 'Rhm': [], 'Rcm': [],
                            'kT': [], 'Qhm': [], 'qhmplot': [], 'Tchm': [], 'Tccm': [], 'deltaTc': [], 'eta': [],
                            'Pelem': [], 'Edshh': [], 'slotf': [], 'nTEMrowcur': [], 'rownum': []
                        }
                        nummax = cmcannummax

                        PTEG, PTEGmin, PTEGmax, TcmTEGoutmax, TcmTEGoutmin = 0, 0, 0, 0, 0
                        wcm0sv = wcm0
                        Tcm0sv = Tcm0
                        whm0sv = whm0
                        Thm0sv = Thm0
                        slotf = 1
                        mTcmTEGout = np.zeros(10000)
                        TEM_distrib_TEG_lst = []
                        # расчёт последовательных каналов
                        for num in range(1, nummax+1):
                            mThmin = np.zeros(10000)
                            mThmout = np.zeros(10000)
                            mTcmout = np.zeros(10000)
                            mwhmout = np.zeros(10000)
                            mwcmout = np.zeros(10000)
                            wcmin = wcm0
                            whmin = whm0
                            Tcmin = Tcm0
                            Thmin = Thm0

                            # конфигурация ТЭС для текущего ОТЭГ
                            can_idx = np.round((num - 1) * 2 / (nummax - 1)).astype(int)
                            mTEMisp, nTEMrowrezm = layer_distr_func(
                                distr_perc=tem_distr[can_idx], rownumtot=rownumtot1,
                                nTEMrow=nTEMrowhm, Lpolez=Lpolez
                            )
                            TEM_distrib_TEG_lst.append(mTEMisp)

                            PTEGcan, teplsuc1, rownum, imas, imas1, imas1sl, TEMnum1 = 0, 0, 1, 1, 1, 1, nTEMrowrezm[1]
                            # тепловой расчёт канала
                            while teplsuc1 == 0:
                                # расчётный слот заполнен
                                if slotf == 1:
                                    # число пустых слотов вокруг расчётного ТЭМ
                                    TEMqua, TEMpr1, TEMpr2 = 0, 0, 0
                                    if imas1sl > 1:
                                        imas2 = imas1sl - 1
                                    for jm in range(1, nTEMrowhm + 1):
                                        masparam = mTEMisp[rownum, jm]
                                        if masparam == 1:
                                            TEMqua += 1
                                        if imas1sl == 1:
                                            if masparam == 0 and TEMqua == imas1sl:
                                                TEMpr2 += 1
                                        if imas1sl == TEMnum1:
                                            if masparam == 0 and TEMqua == imas2:
                                                TEMpr1 += 1
                                        if imas1sl > 1 and imas1sl < TEMnum1:
                                            if masparam == 0 and TEMqua == imas2:
                                                TEMpr1 += 1
                                            if masparam == 0 and TEMqua == imas1sl:
                                                TEMpr2 += 1
                                    if imas1sl == 1:
                                        nslot = 1 + TEMpr2/2
                                    if imas1sl == TEMnum1:
                                        nslot = 1 + TEMpr1/2
                                    if imas1sl > 1 and imas1sl < TEMnum1:
                                        nslot = 1 + TEMpr1/2 + TEMpr2/2

                                    Arhm1 = Arrowhm*nslot/(2*nNTEG*nTEMrowhm)
                                    Arthmelem = Arthm*nslot/(2*nNTEG*nTEMrowhm*rownumtot1)
                                    Arcm1 = Arrowcm/(2*nNTEG*rownumtot1)
                                    Artcmelem = Artcm*nslot/(2*nNTEG*nTEMrowhm*rownumtot1)
                                    Artosn = Lhm*Lcm*nslot/(nTEMrowhm*rownumtot1)
                                    Thm = Thmin
                                    Tcm = Tcmin
                                    whm = whmin
                                    wcm = wcmin
                                    if fin_type == 'Continuous':
                                        Tplhm = Thm - (Thm - Tcm)/4
                                        Tsthm = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                            ((Thm + Tplhm)/2)*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                                    Tplcm = Tcm + (Thm - Tcm)/4
                                    Tstcm = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                                        ((Tcm + Tplcm)/2)*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                                    dTout1suc = 0
                                    Thmout = Thmin
                                    Tcmout = Tcmin

                                    # определение температур на выходе в режиме ХХ
                                    while dTout1suc == 0:
                                        rohmm = rohm0*Thm0/Thm
                                        nuhmm = nuex(Thm, rH2O)
                                        lbdhmm = lbdex(Thm, rH2O)
                                        chmm = cpex(Thm, rH2O, lbdhmm, rohmm, nuhmm)
                                        Prhmm = nuhmm*rohmm*chmm/lbdhmm
                                        Reyhm = whm*dhidrhm/nuhmm

                                        # сплошное оребрение
                                        if fin_type == 'Continuous':
                                            if Reyhm <= 2300:
                                                Tstsuc = 0
                                                # уточнение осреднённой температуры металла между оребрением и основанием
                                                while Tstsuc == 0:
                                                    rohmst = rohm0*Thm0/Tsthm
                                                    nuhmst = nuex(Tsthm, rH2O)
                                                    alphateplhm = 1.86 * (Reyhm*Prhmm/(Lcm/dhidrhm))**(1/3) * (rohmm*nuhmm/(rohmst*nuhmst))**0.14 * lbdhmm/dhidrhm
                                                    mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5 / deltahm
                                                    Tfinhm = Thm - (Thm - Tplhm)*math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                                    Tsthm1 = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                                        Tfinhm*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                                                    delTst = abs(Tsthm - Tsthm1)
                                                    delTst1 = Tsthm - Tsthm1
                                                    if delTst > 0.1:
                                                        Tsthm = Tsthm - 0.05 if delTst1 > 0 else Tsthm + 0.05
                                                    else:
                                                        Tstsuc = 1
                                            else:
                                                fric = 1 / (1.58*math.log(Reyhm) - 3.28)**2
                                                alphateplhm = ((fric/2)*(Reyhm - 1000)*Prhmm/(1 + 12.7*(fric/2)**0.5*(Prhmm**(2/3) - 1)))*lbdhmm/dhidrhm
                                                mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5/deltahm
                                            etafhm = math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                            etafhmtot = 1 - Arfhm*(1 - etafhm)/Arthm
                                        # рассечное оребрение
                                        else:
                                            if Reyhm <= ReyNus:
                                                Nushm = 4.37*0.0001*(deltahm/dhidrhm)**(-2.6)*(ledhm/dhidrhm)**(-0.15)*Reyhm**(2.2*(deltahm/dhidrhm)**0.55*(ledhm/dhidrhm)**(-0.02))
                                            else:
                                                Nushm = 7.23*0.001*(deltahm/dhidrhm)**(-1.6)*(ledhm/dhidrhm)**(-0.9)*Reyhm**(1.2*(deltahm/dhidrhm)**0.34*(ledhm/dhidrhm)**0.15)
                                            alphateplhm = Nushm*lbdhmm/dhidrhm
                                            mfhm = (2*alphateplhm*deltahm/lbdNTEG)**0.5/deltahm
                                            etafhmtot = 1
                                        alphateplfhm = alphateplhm*etafhmtot

                                        rocmm = row(Tcm)
                                        nucmm = nuw(Tcm, rocmm)
                                        lbdcmm = lbdw(Tcm)
                                        ccmm = cpw(Tcm)
                                        Prcmm = ccmm * rocmm * nucmm / lbdcmm
                                        Reycm = wcm * dhidrcm / nucmm
                                        if Reycm <= 2300:
                                            Tstsuc = 0
                                            # уточнение осреднённой температуры металла между оребрением и основанием
                                            while Tstsuc == 0:
                                                rocmst = row(Tstcm)
                                                nucmst = nuw(Tstcm, rocmm)
                                                alphateplcm = 1.86 * (Reycm*Prcmm/(Lhm/dhidrcm))**(1/3) * (rocmm*nucmm/(rocmst*nucmst))**0.14 * lbdcmm/dhidrcm
                                                mfcm = (2*alphateplcm*deltacm/lbdOTEG)**0.5 / deltacm
                                                Tfincm = Tcm + (Tplcm - Tcm)*math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                                                Tstcm1 = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                                                    Tfincm*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                                                delTst = abs(Tstcm - Tstcm1)
                                                delTst1 = Tstcm - Tstcm1
                                                if delTst > 0.1:
                                                    Tstcm = Tstcm - 0.05 if delTst1 > 0 else Tstcm + 0.05
                                                else:
                                                    Tstsuc = 1
                                        else:
                                            fric = 1 / (1.58*math.log(Reycm) - 3.28)**2
                                            alphateplcm = ((fric/2)*(Reycm - 1000)*Prcmm/(1 + 12.7*(fric/2)**0.5*(Prcmm**(2/3) - 1))) * lbdcmm/dhidrcm
                                            mfcm = (2*alphateplcm*deltacm/lbdOTEG)**0.5/deltacm
                                        etafcm = math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                                        etafcmtot = 1 - Arfcm*(1 - etafcm)/Artcm
                                        alphateplfcm = alphateplcm*etafcmtot
                                        UA = 1/(
                                            (1/(kkonvhm*alphateplfhm*Arthmelem)) + (bsthichm/(lbdNTEG*Artosn)) + (hTEM/(lbd0*nTE*ArTE)) + \
                                            2*(0.0001/(ktp*0.8*Arcer)) + (bsthiccm/(lbdOTEG*Artosn)) + (1/(kkonvcm*alphateplfcm*Artcmelem))
                                        )
                                        temkGhm = Arhm1*whm*rohmm*chmm
                                        temkGcm = Arcm1*wcm*rocmm*ccmm
                                        if temkGhm < temkGcm:
                                            temkGmin = temkGhm
                                            Cr = temkGhm/temkGcm
                                        else:
                                            temkGmin = temkGcm
                                            Cr = temkGcm/temkGhm
                                        NTU = UA/temkGmin
                                        # обе среды не перемешиваются
                                        if Reyhm < 2300:
                                            eps = 1 - math.exp(NTU**0.22*(math.exp(-1*Cr*NTU**0.78) - 1)/Cr)
                                        else:  # перемешивается горячая среда
                                            if temkGhm < temkGcm:
                                                eps = 1 - math.exp((-1/Cr)*(1 - math.exp(-1*Cr*NTU)))
                                            else:
                                                eps = (1/Cr)*(1 - math.exp(-1*Cr*(1 - math.exp(-1*NTU))))
                                        Qhm = eps*temkGmin*(Thmin - Tcmin)*kiz
                                        Thmout1 = Thmin - eps*(temkGmin/temkGhm)*(Thmin - Tcmin)
                                        Tcmout1 = Tcmin + eps*(temkGmin/temkGcm)*(Thmin - Tcmin) * kiz
                                        delThmout = abs(Thmout - Thmout1)
                                        delTcmout = abs(Tcmout - Tcmout1)
                                        if delThmout > 0.1 or delTcmout > 0.1:
                                            if delThmout > 0.1:
                                                delThmout1 = Thmout - Thmout1
                                                Thmout = Thmout - 0.01 if delThmout1 > 0 else Thmout + 0.01
                                                Thm = (Thmin + Thmout) / 2
                                            if delTcmout > 0.1:
                                                delTcmout1 = Tcmout - Tcmout1
                                                Tcmout = Tcmout - 0.01 if delTcmout1 > 0 else Tcmout + 0.01
                                                Tcm = (Tcmin + Tcmout)/2
                                            whmout = Vhm * Thmout / (Thm0*Arrowhm)
                                            whm = (whmin + whmout)/2
                                            rocmout = row(Tcmout)
                                            wcmout = wcm0*rocm0/rocmout
                                            wcm = (wcmin + wcmout)/2
                                            if fin_type == 'Continuous' and Reyhm <= 2300:
                                                Tplhm = Thm - Qhm/(alphateplfhm*Arthmelem)
                                                Tfinhm = Thm - (Thm - Tplhm)*math.tanh(mfhm*((hhm/2) - deltahm))/(mfhm*((hhm/2) - deltahm))
                                                Tsthm = Tplhm*((shaghm - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm))) + \
                                                    Tfinhm*(2*((hhm/2) - deltahm)/((shaghm - deltahm) + 2*((hhm/2) - deltahm)))
                                            if Reycm <= 2300:
                                                Tplcm = Tcm + Qhm/(alphateplfcm*Artcmelem)
                                                Tfincm = Tcm + (Tplcm - Tcm)*math.tanh(mfcm*((hcmm/2) - deltacm))/(mfcm*((hcmm/2) - deltacm))
                                                Tstcm = Tplcm*((shagcm - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm))) + \
                                                    Tfincm*(2*((hcmm/2) - deltacm)/((shagcm - deltacm) + 2*((hcmm/2) - deltacm)))
                                        else:
                                            dTout1suc = 1

                                    Rhm = nTE*ArTE/(kkonvhm*alphateplfhm*Arthmelem) + bsthichm*nTE*ArTE/(lbdNTEG*Artosn) + 0.0001*nTE*ArTE/(ktp*0.8*Arcer)
                                    Rcm = nTE*ArTE/(kkonvcm*alphateplfcm*Artcmelem) + bsthiccm*nTE*ArTE/(lbdOTEG*Artosn) + 0.0001*nTE*ArTE/(ktp*0.8*Arcer)
                                    Tchm = Thm - Qhm*Rhm/(nTE*ArTE)
                                    Tccm = Tcm + Qhm*Rcm/(nTE*ArTE)
                                    kT = 1 + Z0*((Tchm + Tccm)/2 + m*Tchm)/(1 + m)**2
                                    Edshh = nTE*alpha0*(Tchm - Tccm)

                                    dTout2suc = 0
                                    # итерационные уточнения генераторного режима
                                    while dTout2suc == 0:
                                        deltaTc = Tchm - Tccm
                                        eta = Z0*deltaTc*m/(kT*(1 + m)**2)
                                        qhmplot = deltaTc*kT/R
                                        Qhm = qhmplot*nTE*ArTE
                                        Pelem = Qhm*eta
                                        Thmout = Thmin - Qhm/(kiz*temkGhm)
                                        Thm = (Thmin + Thmout)/2
                                        Tchm1 = Thm - qhmplot*Rhm
                                        Qcm = Qhm*(1 - eta)
                                        Tcmout = Tcmin + Qcm/temkGcm
                                        Tcm = (Tcmin + Tcmout)/2
                                        Tccm1 = Tcm + qhmplot*(1 - eta)*Rcm
                                        delTchm = abs(Tchm - Tchm1)
                                        delTccm = abs(Tccm - Tccm1)
                                        if delTchm > 0.1 or delTccm > 0.1:
                                            if delTchm > 0.1:
                                                delTchm1 = Tchm - Tchm1
                                                Tchm = Tchm - 0.01 if delTchm1 > 0 else Tchm + 0.01
                                            if delTccm > 0.1:
                                                delTccm1 = Tccm - Tccm1
                                                Tccm = Tccm - 0.01 if delTccm1 > 0 else Tccm + 0.01
                                            kT = 1 + Z0*((Tchm + Tccm)/2 + m*Tchm)/(1 + m)**2
                                        else:
                                            dTout2suc = 1

                                    whmout = Vhm*Thmout/(Thm0*Arrowhm)
                                    whm = (whmin + whmout)/2
                                    rocmout = row(Tcmout)
                                    wcmout = wcm0*rocm0/rocmout
                                    wcm = (wcmin + wcmout)/2
                                    Tplhm = Thm - Qhm/(alphateplfhm*Arthmelem)
                                    Tplcm = Tcm + Qcm/(alphateplfcm*Artcmelem)

                                    mwhmout[imas], mThmin[imas], mThmout[imas], mTcmout[imas], mwcmout[imas] = whmout, Thmin, Thmout, Tcmout, wcmout
                                    PTEGcan += Pelem
                                    imas1sl += 1
                                # пустой слот
                                else:
                                    mwhmout[imas], mThmin[imas], mThmout[imas], mTcmout[imas], mwcmout[imas] = 0, 0, 0, 0, 0
                                    (
                                        Reycm, alphateplcm, alphateplfcm, Prcmm, lbdcmm, rocmm, ccmm, nucmm, wcm,
                                        Tcm, Tcmin, Tplcm, temkGcm, etafcmtot, Reyhm, alphateplhm, alphateplfhm,
                                        Prhmm, lbdhmm, rohmm, chmm, nuhmm, whm, Thm, Thmin, Thmout, Tplhm,
                                        temkGhm, etafhmtot, UA, NTU, eps, Rhm, Rcm, kT, Qhm, qhmplot,
                                        Tchm, Tccm, deltaTc, eta, Pelem, Edshh
                                    ) = (
                                        0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0
                                    )
                                for key in params_dict.keys():
                                    if key not in ['Tcmout', 'slotf', 'nTEMrowcur']:
                                        params_dict[key].append(locals()[key])
                                params_dict['Tcmout'].append(mTcmout[imas])
                                params_dict['slotf'].append(mTEMisp[rownum, imas1])
                                params_dict['nTEMrowcur'].append(nTEMrowrezm[rownum])

                                imas += 1
                                imas1 += 1

                                # переход к следующему ряду
                                if imas1 > nTEMrowhm:
                                    rownum += 1
                                    if rownum > rownumtot1:
                                        teplsuc1 = 1
                                    else:
                                        imas1 = 1
                                        imas1sl = 1
                                        TEMnum1 = nTEMrowrezm[rownum]
                                        svin = nTEMrowrezm[rownum-1]
                                # переход к учатску, расположенному левее
                                if teplsuc1 == 0:
                                    slotf = mTEMisp[rownum, imas1]
                                    # в следующем слоте есть ТЭМ
                                    if slotf == 1:
                                        # параметры воды для первого участка
                                        if imas1 == 1:
                                            Tcmin = Tcm0
                                            wcmin = wcm0
                                        # параметры воды для участков левее
                                        else:
                                            Tcmin = Tcmout
                                            wcmin = wcmout
                                        # параметры горячей среды для первого ряда
                                        if rownum == 1:
                                            whmin = whm0
                                            Thmin = Thm0
                                        # параметры горячей среды в последующих рядах
                                        else:
                                            TEMqua, slot1, slot2, chon, slotnum = 0, 0, 0, 0, 0
                                            # наличие пустых слотов на границе участка
                                            while chon == 0:
                                                slotnum += 1
                                                masparam = mTEMisp[rownum, slotnum]
                                                if masparam == 1:
                                                    TEMqua += 1
                                                if TEMqua == imas1sl:
                                                    chon = 1
                                                    if imas1sl == 1:
                                                        slot2 = mTEMisp[rownum, slotnum+1];
                                                    if imas1sl == TEMnum1:
                                                        slot1 = mTEMisp[rownum, slotnum-1];
                                                    if imas1sl > 1 and imas1sl < TEMnum1:
                                                        slot1, slot2 = mTEMisp[rownum, slotnum-1], mTEMisp[rownum, slotnum+1]

                                            TEMqua, TEMpr1, TEMpr2, slotnum, slotnum1, slotnum2, kisp1, kisp2, TEMslot = 0, 0, 0, 0, 0, 0, 0, 0, 0
                                            # длины промежутков, граничные номера
                                            for jm in range(1, nTEMrowhm + 1):
                                                slotnum += 1
                                                masparam = mTEMisp[rownum, jm]
                                                if masparam == 1:
                                                    TEMqua += 1
                                                if masparam == 1 and TEMqua == imas1sl:
                                                    TEMslot = slotnum

                                                if imas1sl == 1:
                                                    slotnum1 = 1
                                                    if slot2 == 0:
                                                        masparam = mTEMisp[rownum, jm]
                                                        if masparam == 0 and TEMqua == imas1sl:
                                                            TEMpr2 += 1
                                                            slotnum2 = slotnum
                                                    else:
                                                        slotnum2 = 1

                                                if imas1sl == TEMnum1:
                                                    slotnum2 = nTEMrowhm
                                                    if slot1 == 0:
                                                        imas2 = imas1sl - 1
                                                        masparam = mTEMisp[rownum, jm]
                                                        if masparam == 0 and TEMqua == imas2:
                                                            TEMpr1 += 1
                                                            if TEMpr1 == 1:
                                                                slotnum1 = slotnum
                                                    else:
                                                        slotnum1 = nTEMrowhm

                                                if imas1sl > 1 and imas1sl < TEMnum1:
                                                    if slot1 == 0:
                                                        imas2 = imas1sl - 1
                                                        masparam = mTEMisp[rownum, jm]
                                                        if masparam == 0 and TEMqua == imas2:
                                                            TEMpr1 += 1
                                                            if TEMpr1 == 1:
                                                                slotnum1 = slotnum
                                                    else:
                                                        slotnum1 = TEMslot
                                                    if slot2 == 0:
                                                        masparam = mTEMisp[rownum, jm];
                                                        if masparam == 0 and TEMqua == imas1sl:
                                                            TEMpr2 += 1
                                                            slotnum2 = slotnum
                                                    else:
                                                        slotnum2 = TEMslot

                                            # определение коэффициентов использования и крайних слотов
                                            if imas1sl == 1:
                                                kisp1 = 1
                                                if slot2 == 0:
                                                    TEMpr2 /= 2
                                                    TEMpr21 = TEMpr2 - math.trunc(TEMpr2)
                                                    if TEMpr21 == 0:
                                                        slotnum2 -= round(TEMpr2)
                                                        kisp2 = 1
                                                    else:
                                                        slotnum2 -= math.trunc(TEMpr2)
                                                        kisp2 = 0
                                                else:
                                                    kisp2 = 1

                                            if imas1sl == TEMnum1:
                                                kisp2 = 1
                                                if slot1 == 0:
                                                    TEMpr1 /= 2
                                                    TEMpr11 = TEMpr1 - math.trunc(TEMpr1)
                                                    if TEMpr11 == 0:
                                                        slotnum1 += round(TEMpr1)
                                                        kisp1 = 1
                                                    else:
                                                        slotnum1 += math.trunc(TEMpr1)
                                                        kisp1 = 0
                                                else:
                                                    kisp1 = 1

                                            if imas1sl > 1 and imas1sl < TEMnum1:
                                                if slot1 == 0:
                                                    TEMpr1 /= 2
                                                    TEMpr11 = TEMpr1 - math.trunc(TEMpr1)
                                                    if TEMpr11 == 0:
                                                        slotnum1 += round(TEMpr1)
                                                        kisp1 = 1
                                                    else:
                                                        slotnum1 += math.trunc(TEMpr1)
                                                        kisp1 = 0
                                                else:
                                                    kisp1 = 1
                                                if slot2 == 0:
                                                    TEMpr2 /= 2
                                                    TEMpr21 = TEMpr2 - math.trunc(TEMpr2)
                                                    if TEMpr21 == 0:
                                                        slotnum2 -= round(TEMpr2)
                                                        kisp2 = 1
                                                    else:
                                                        slotnum2 -= math.trunc(TEMpr2)
                                                        kisp2 = 0
                                                else:
                                                    kisp2 = 1

                                            if kisp1 == 1 and kisp2 == 1:
                                                kispcent = 1/(slotnum2 - slotnum1 + 1)
                                                kispkr1 = kispcent
                                                kispkr2 = kispcent
                                            if kisp1 == 0 and kisp2 == 0:
                                                kispcent = 1/(slotnum2 - slotnum1)
                                                kispkr1 = kispcent/2
                                                kispkr2 = kispkr1
                                            if kisp1 == 0 and kisp2 == 1:
                                                kispcent = 1/(slotnum2 - slotnum1 + 0.5)
                                                kispkr1 = kispcent/2
                                                kispkr2 = kispcent
                                            if kisp1 == 1 and kisp2 == 0:
                                                kispcent = 1/(slotnum2 - slotnum1 + 0.5)
                                                kispkr1 = kispcent
                                                kispkr2 = kispcent/2

                                            slotnum, whmin, Thmin = slotnum1, 0, 0
                                            # средние параметры
                                            while slotnum <= slotnum2:
                                                if slotnum == 1:
                                                    slotnumpr1 = slotnum
                                                    slotnumpr2 = slotnum + 1
                                                    slotnumpr3 = slotnum + 2
                                                if slotnum == nTEMrowhm:
                                                    slotnumpr1 = slotnum - 2
                                                    slotnumpr2 = slotnum - 1
                                                    slotnumpr3 = slotnum
                                                if slotnum > 1 and slotnum < nTEMrowhm:
                                                    slotnumpr1 = slotnum - 1
                                                    slotnumpr2 = slotnum
                                                    slotnumpr3 = slotnum + 1

                                                # для первого из трёх слотов
                                                if slotnumpr1 == 1:
                                                    whmoutsl1 = mwhmout[imas - imas1 - nTEMrowhm + 1]
                                                    Thmoutsl1 = mThmout[imas - imas1 - nTEMrowhm + 1]
                                                else:
                                                    slotnumpr = 0
                                                    for jm in range(1, nTEMrowhm + 1):
                                                        slotnumpr += 1
                                                        masparam = mTEMisp[rownum - 1, jm]
                                                        if masparam == 1:
                                                            TEMslotpr = slotnumpr
                                                        # в слоте находится ТЭМ
                                                        if masparam == 1 and slotnumpr == slotnumpr1:
                                                            whmoutsl1 = mwhmout[imas - imas1 - nTEMrowhm + slotnumpr]
                                                            Thmoutsl1 = mThmout[imas - imas1 - nTEMrowhm + slotnumpr]
                                                        # пустой слот
                                                        if masparam == 0 and slotnumpr == slotnumpr1:
                                                            slotquapr, TEMquapr1 = 0, 0
                                                            for km in range(TEMslotpr, nTEMrowhm + 1):
                                                                if TEMquapr1 < 2:
                                                                    masparam = mTEMisp[rownum - 1, km]
                                                                    if masparam == 0:
                                                                        slotquapr += 1
                                                                    else:
                                                                        TEMquapr1 += 1
                                                            slotquapr1 = slotquapr/2 - math.trunc(slotquapr/2)
                                                            if slotquapr == 1:
                                                                slotnumch = TEMslotpr + 1
                                                            else:
                                                                slotnumch = TEMslotpr + round(slotquapr/2)
                                                            if (slotnumpr1 < slotnumch) or (slotnumpr1 == slotnumch and slotquapr1 == 0):  # слот относится к предыдущему ТЭМ
                                                                whmoutsl1 = mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr]
                                                                Thmoutsl1 = mThmout[imas - imas1 - nTEMrowhm + TEMslotpr]
                                                            if slotnumpr1 > slotnumch:  # слот относится к следующему ТЭМ
                                                                whmoutsl1 = mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                                Thmoutsl1 = mThmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                            if slotnumpr1 == slotnumch and slotquapr1 > 0:  # слот относится к обоим ТЭМ
                                                                whmoutsl1 = (
                                                                    mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr] + \
                                                                    mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                                )/2
                                                                Thmoutsl1 = (
                                                                    mThmout[imas - imas1 - nTEMrowhm + TEMslotpr] + \
                                                                    mThmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                                )/2

                                                # для второго из трёх слотов
                                                slotnumpr = 0
                                                for jm in range(1, nTEMrowhm + 1):
                                                    slotnumpr += 1
                                                    masparam = mTEMisp[rownum - 1, jm]
                                                    if masparam == 1:
                                                        TEMslotpr = slotnumpr
                                                    # в слоте находится ТЭМ
                                                    if masparam == 1 and slotnumpr == slotnumpr2:
                                                        whmoutsl2 = mwhmout[imas - imas1 - nTEMrowhm + slotnumpr]
                                                        Thmoutsl2 = mThmout[imas - imas1 - nTEMrowhm + slotnumpr]
                                                    # пустой слот
                                                    if masparam == 0 and slotnumpr == slotnumpr2:
                                                        slotquapr, TEMquapr1 = 0, 0
                                                        for km in range(TEMslotpr, nTEMrowhm + 1):
                                                            if TEMquapr1 < 2:
                                                                masparam = mTEMisp[rownum - 1, km]
                                                                if masparam == 0:
                                                                    slotquapr += 1
                                                                else:
                                                                    TEMquapr1 += 1
                                                        slotquapr1 = slotquapr/2 - math.trunc(slotquapr/2)
                                                        if slotquapr == 1:
                                                            slotnumch = TEMslotpr + 1
                                                        else:
                                                            slotnumch = TEMslotpr + round(slotquapr/2)
                                                        if (slotnumpr2 < slotnumch) or (slotnumpr2 == slotnumch and slotquapr1 == 0):  # слот относится к предыдущему ТЭМ
                                                            whmoutsl2 = mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr]
                                                            Thmoutsl2 = mThmout[imas - imas1 - nTEMrowhm + TEMslotpr]
                                                        if slotnumpr2 > slotnumch:  # слот относится к следующему ТЭМ
                                                            whmoutsl2 = mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                            Thmoutsl2 = mThmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                        if slotnumpr2 == slotnumch and slotquapr1 > 0:  # слот относится к обоим ТЭМ
                                                            whmoutsl2 = (
                                                                mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr] + \
                                                                mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                            )/2
                                                            Thmoutsl2 = (
                                                                mThmout[imas - imas1 - nTEMrowhm + TEMslotpr] + \
                                                                mThmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                            ) / 2

                                                # для третьего из трёх слотов
                                                if slotnumpr3 == nTEMrowhm:
                                                    whmoutsl3 = mwhmout[imas - imas1]
                                                    Thmoutsl3 = mThmout[imas - imas1]
                                                else:
                                                    slotnumpr = 0
                                                    for jm in range(1, nTEMrowhm + 1):
                                                        slotnumpr += 1
                                                        masparam = mTEMisp[rownum - 1, jm]
                                                        if masparam == 1:
                                                            TEMslotpr = slotnumpr
                                                        if masparam == 1 and slotnumpr == slotnumpr3:  # в слоте находится ТЭМ
                                                            whmoutsl3 = mwhmout[imas - imas1 - nTEMrowhm + slotnumpr]
                                                            Thmoutsl3 = mThmout[imas - imas1 - nTEMrowhm + slotnumpr]
                                                        if masparam == 0 and slotnumpr == slotnumpr3:  # пустой слот
                                                            slotquapr, TEMquapr1 = 0, 0
                                                            for km in range(TEMslotpr, nTEMrowhm + 1):
                                                                if TEMquapr1 < 2:
                                                                    masparam = mTEMisp[rownum - 1, km]
                                                                    if masparam == 0:
                                                                        slotquapr += 1
                                                                    else:
                                                                        TEMquapr1 += 1
                                                            slotquapr1 = slotquapr/2 - math.trunc(slotquapr/2)
                                                            if slotquapr == 1:
                                                                slotnumch = TEMslotpr + 1
                                                            else:
                                                                slotnumch = TEMslotpr + round(slotquapr/2)
                                                            if (slotnumpr3 < slotnumch) or (slotnumpr3 == slotnumch and slotquapr1 == 0):  # слот относится к предыдущему ТЭМ
                                                                whmoutsl3 = mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr]
                                                                Thmoutsl3 = mThmout[imas - imas1 - nTEMrowhm + TEMslotpr]
                                                            if slotnumpr3 > slotnumch:  # слот относится к следующему ТЭМ
                                                                whmoutsl3 = mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                                Thmoutsl3 = mThmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                            if slotnumpr3 == slotnumch and slotquapr1 > 0:  # слот относится к обоим ТЭМ
                                                                whmoutsl3 = (
                                                                    mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr] + \
                                                                    mwhmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                                )/2
                                                                Thmoutsl3 = (
                                                                    mThmout[imas - imas1 - nTEMrowhm + TEMslotpr] + \
                                                                    mThmout[imas - imas1 - nTEMrowhm + TEMslotpr + slotquapr + 1]
                                                                )/2

                                                if slotnum == slotnum1:
                                                    whmin = kispkr1*(whmoutsl1 + whmoutsl2 + whmoutsl3)/3
                                                    Thmin = kispkr1*(Thmoutsl1 + Thmoutsl2 + Thmoutsl3)/3
                                                if slotnum == slotnum2 and slotnum2 > slotnum1:
                                                    whmin += kispkr2*(whmoutsl1 + whmoutsl2 + whmoutsl3)/3
                                                    Thmin += kispkr2*(Thmoutsl1 + Thmoutsl2 + Thmoutsl3)/3
                                                if slotnum > slotnum1 and slotnum < slotnum2:
                                                    whmin += kispcent*(whmoutsl1 + whmoutsl2 + whmoutsl3)/3
                                                    Thmin += kispcent*(Thmoutsl1 + Thmoutsl2 + Thmoutsl3)/3

                                                slotnum += 1
                            PTEG += 2*PTEGcan
                            wcm0, Tcm0 = 0, 0
                            # средние параметры на выходе из канала
                            for im in range(1, rownumtot1+1):
                                Tcm0 += mTcmout[nTEMrowhm*im]
                                wcm0 += mwcmout[nTEMrowhm*im]
                            Tcm0 = Tcm0/rownumtot1
                            wcm0 = wcm0/rownumtot1
                            mTcmTEGout[num] = Tcm0
                            if num == cmcannummin:
                                PTEGmin = PTEG
                                TcmTEGoutmin = Tcm0
                            if num == cmcannummax:
                                PTEGmax = PTEG
                                TcmTEGoutmax = Tcm0

                        PTEG = cmgrmax*PTEGmax + cmgrmin*PTEGmin
                        Qcm = cmflownum*Arrowcm*wcm0sv/nNTEG
                        Nrtccm = Qcm*rtccm/0.75
                        Nrtchm = 0
                        Pus = PTEG - Nrtchm - Nrtccm
                        results_dict['message'] = 'Расчёт выполнен'
                        results_dict['Pus'] = Pus

                        # Полный расчёт конструктивных и режимных параметров
                        if full_design:
                            TcmTEGout = (cmgrmax / cmflownum) * TcmTEGoutmax + (cmgrmin / cmflownum) * TcmTEGoutmin
                            TEM_distrib_TEG = np.array(TEM_distrib_TEG_lst)
                            TEMlayq = round(nTEMrowhm*rownumtot1)
                            TEMlayq1 = round(TEMlayq*cmcannummax)
                            TEMlayq2 = nTEMrowrezm[1:rownumtot1+1].sum()

                            params_df = pd.DataFrame(params_dict)
                            (
                                Reycm, alphateplfcm, Prcmm, lbdcmm, rocmm, cpcmm, nucmm, wcm, Tcm, temkGcm,
                                Reyhm, alphateplfhm, Prhmm, lbdhmm, rohmm, cphmm, nuhmm, whm, Thm, temkGhm,
                                ThmTEGout, UA, NTU, eps, Rhm, Rcm, kT,
                                Qhm, qhmplot, Tchm, Tccm, deltaTc, eta, Pelem,
                            ) = (
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0,
                            )

                            ifin, im, imas, jmas = TEMlayq1, 0, 1, 1
                            (
                                mReycm, malphateplfcm, mPrcmm, mlbdcmm, mrocmm, mcpcmm, mnucmm, mwcm, mTcm, mtemkGcm,
                                mReyhm, malphateplfhm, mPrhmm, mlbdhmm, mrohmm, mcphmm, mnuhmm, mwhm, mThm, mtemkGhm,
                                mUA, mNTU, meps, mRhm, mRcm, mkT, mQhm, mqhmplot, mTchm, mTccm, mdeltaTc, meta, mPelem,
                                mThmTEGout,
                                mCReycm, mCalphateplfcm, mCPrcmm, mClbdcmm, mCrocmm, mCcpcmm, mCnucmm, mCwcm, mCTcm, mCtemkGcm,
                                mCReyhm, mCalphateplfhm, mCPrhmm, mClbdhmm, mCrohmm, mCcphmm, mCnuhmm, mCwhm, mCThm, mCtemkGhm,
                                mCUA, mCNTU, mCeps, mCRhm, mCRcm, mCkT, mCQhm, mCqhmplot, mCTchm, mCTccm, mCdeltaTc, mCeta, mCPelem
                            ) = (
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
                                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000)
                            )
                            # переход к следующему ТЭМ - следующей строке
                            for _, result_row in params_df.iterrows():
                                im += 1
                                # чтение строки
                                (
                                    Reycm2, alphateplfcm2, Prcmm2, lbdcmm2, rocmm2, cpcmm2, nucmm2, wcm2, Tcm2, temkGcm2,
                                    Reyhm2, alphateplfhm2, Prhmm2, lbdhmm2, rohmm2, cphmm2, nuhmm2, whm2, Thm2, temkGhm2,
                                    UA2, NTU2, eps2, Rhm2, Rcm2, kT2, Qhm2, qhmplot2, Tchm2, Tccm2, deltaTc2, eta2, Pelem2
                                ) = (
                                    result_row['Reycm'], result_row['alphateplfcm'], result_row['Prcmm'], result_row['lbdcmm'], result_row['rocmm'],
                                    result_row['ccmm'], result_row['nucmm'], result_row['wcm'], result_row['Tcm'], result_row['temkGcm'],
                                    result_row['Reyhm'], result_row['alphateplfhm'], result_row['Prhmm'], result_row['lbdhmm'], result_row['rohmm'],
                                    result_row['chmm'], result_row['nuhmm'], result_row['whm'], result_row['Thm'], result_row['temkGhm'],
                                    result_row['UA'], result_row['NTU'], result_row['eps'], result_row['Rhm'], result_row['Rcm'], result_row['kT'],
                                    result_row['Qhm'], result_row['qhmplot'], result_row['Tchm'], result_row['Tccm'], result_row['deltaTc'],
                                    result_row['eta'], result_row['Pelem']
                                )
                                if imas == rownumtot1:
                                    ThmTEGout2 = result_row['Thmout']

                                # осреднение параметров для ТЭГ
                                if jmas <= nummax:
                                    # осреднение параметров для канала/слоя
                                    if imas <= rownumtot1:
                                        # осреднение параметров для ряда
                                        if im <= nTEMrowhm:
                                            Reycm += Reycm2
                                            alphateplfcm += alphateplfcm2
                                            Prcmm += Prcmm2
                                            lbdcmm += lbdcmm2
                                            rocmm += rocmm2
                                            cpcmm += cpcmm2
                                            nucmm += nucmm2
                                            wcm += wcm2
                                            Tcm += Tcm2
                                            temkGcm += temkGcm2
                                            Reyhm += Reyhm2
                                            alphateplfhm += alphateplfhm2
                                            Prhmm += Prhmm2
                                            lbdhmm += lbdhmm2
                                            rohmm += rohmm2
                                            cphmm += cphmm2
                                            nuhmm += nuhmm2
                                            whm += whm2
                                            Thm += Thm2
                                            temkGhm += temkGhm2
                                            UA += UA2
                                            NTU += NTU2
                                            eps += eps2
                                            Rhm += Rhm2
                                            Rcm += Rcm2
                                            kT += kT2
                                            Qhm += Qhm2
                                            qhmplot += qhmplot2
                                            Tchm += Tchm2
                                            Tccm += Tccm2
                                            deltaTc += deltaTc2
                                            eta += eta2
                                            Pelem += Pelem2
                                            if imas == rownumtot1:
                                                ThmTEGout += ThmTEGout2

                                        # осреднение параметров для ряда
                                        if im == nTEMrowhm:
                                            mReycm[imas] = Reycm/nTEMrowrezm[imas]
                                            malphateplfcm[imas] = alphateplfcm/nTEMrowrezm[imas]
                                            mPrcmm[imas] = Prcmm/nTEMrowrezm[imas]
                                            mlbdcmm[imas] = lbdcmm/nTEMrowrezm[imas]
                                            mrocmm[imas] = rocmm/nTEMrowrezm[imas]
                                            mcpcmm[imas] = cpcmm/nTEMrowrezm[imas]
                                            mnucmm[imas] = nucmm/nTEMrowrezm[imas]
                                            mwcm[imas] = wcm/nTEMrowrezm[imas]
                                            mTcm[imas] = Tcm/nTEMrowrezm[imas]
                                            mtemkGcm[imas] = temkGcm/nTEMrowrezm[imas]
                                            mReyhm[imas] = Reyhm/nTEMrowrezm[imas]
                                            malphateplfhm[imas] = alphateplfhm/nTEMrowrezm[imas]
                                            mPrhmm[imas] = Prhmm/nTEMrowrezm[imas]
                                            mlbdhmm[imas] = lbdhmm/nTEMrowrezm[imas]
                                            mrohmm[imas] = rohmm/nTEMrowrezm[imas]
                                            mcphmm[imas] = cphmm/nTEMrowrezm[imas]
                                            mnuhmm[imas] = nuhmm/nTEMrowrezm[imas]
                                            mwhm[imas] = whm/nTEMrowrezm[imas]
                                            mThm[imas] = Thm/nTEMrowrezm[imas]
                                            mtemkGhm[imas] = temkGhm/nTEMrowrezm[imas]
                                            mUA[imas] = UA/nTEMrowrezm[imas]
                                            mNTU[imas] = NTU/nTEMrowrezm[imas]
                                            meps[imas] = eps/nTEMrowrezm[imas]
                                            mRhm[imas] = Rhm/nTEMrowrezm[imas]
                                            mRcm[imas] = Rcm/nTEMrowrezm[imas]
                                            mkT[imas] = kT/nTEMrowrezm[imas]
                                            mQhm[imas] = Qhm/nTEMrowrezm[imas]
                                            mqhmplot[imas] = qhmplot/nTEMrowrezm[imas]
                                            mTchm[imas] = Tchm/nTEMrowrezm[imas]
                                            mTccm[imas] = Tccm/nTEMrowrezm[imas]
                                            mdeltaTc[imas] = deltaTc/nTEMrowrezm[imas]
                                            meta[imas] = eta/nTEMrowrezm[imas]
                                            mPelem[imas] = Pelem/nTEMrowrezm[imas]
                                            # средние температуры на выходе из каналов
                                            if imas == rownumtot1:
                                                mThmTEGout[jmas] = ThmTEGout/nTEMrowrezm[imas]
                                                ThmTEGout = 0
                                            (
                                                Reycm, alphateplfcm, Prcmm, lbdcmm, rocmm, cpcmm, nucmm, wcm, Tcm, temkGcm,
                                                Reyhm, alphateplfhm, Prhmm, lbdhmm, rohmm, cphmm, nuhmm, whm, Thm, temkGhm,
                                                ThmTEGout, UA, NTU, eps, Rhm, Rcm, kT,
                                                Qhm, qhmplot, Tchm, Tccm, deltaTc, eta, Pelem,
                                            ) = (
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0,
                                            )
                                            im = 0
                                            imas += 1
                                    # осреднение параметров для канала/слоя
                                    if imas > rownumtot1:
                                        for km in range(1, rownumtot1+1):
                                            Reycm += mReycm[km]
                                            alphateplfcm += malphateplfcm[km]
                                            Prcmm += mPrcmm[km]
                                            lbdcmm += mlbdcmm[km]
                                            rocmm += mrocmm[km]
                                            cpcmm += mcpcmm[km]
                                            nucmm += mnucmm[km]
                                            wcm += mwcm[km]
                                            Tcm += mTcm[km]
                                            temkGcm += mtemkGcm[km]
                                            Reyhm += mReyhm[km]
                                            alphateplfhm += malphateplfhm[km]
                                            Prhmm += mPrhmm[km]
                                            lbdhmm += mlbdhmm[km]
                                            rohmm += mrohmm[km]
                                            cphmm += mcphmm[km]
                                            nuhmm += mnuhmm[km]
                                            whm += mwhm[km]
                                            Thm += mThm[km]
                                            temkGhm += mtemkGhm[km]
                                            UA += mUA[km]
                                            NTU += mNTU[km]
                                            eps += meps[km]
                                            Rhm += mRhm[km]
                                            Rcm += mRcm[km]
                                            kT += mkT[km]
                                            Qhm += mQhm[km]
                                            qhmplot += mqhmplot[km]
                                            Tchm += mTchm[km]
                                            Tccm += mTccm[km]
                                            deltaTc += mdeltaTc[km]
                                            eta += meta[km]
                                            Pelem += mPelem[km]
                                        mCReycm[jmas] = Reycm/rownumtot1
                                        mCalphateplfcm[jmas] = alphateplfcm/rownumtot1
                                        mCPrcmm[jmas] = Prcmm/rownumtot1
                                        mClbdcmm[jmas] = lbdcmm/rownumtot1
                                        mCrocmm[jmas] = rocmm/rownumtot1
                                        mCcpcmm[jmas] = cpcmm/rownumtot1
                                        mCnucmm[jmas] = nucmm/rownumtot1
                                        mCwcm[jmas] = wcm/rownumtot1
                                        mCTcm[jmas] = Tcm/rownumtot1
                                        mCtemkGcm[jmas] = temkGcm/rownumtot1
                                        mCReyhm[jmas] = Reyhm/rownumtot1
                                        mCalphateplfhm[jmas] = alphateplfhm/rownumtot1
                                        mCPrhmm[jmas] = Prhmm/rownumtot1
                                        mClbdhmm[jmas] = lbdhmm/rownumtot1
                                        mCrohmm[jmas] = rohmm/rownumtot1
                                        mCcphmm[jmas] = cphmm/rownumtot1
                                        mCnuhmm[jmas] = nuhmm/rownumtot1
                                        mCwhm[jmas] = whm/rownumtot1
                                        mCThm[jmas] = Thm/rownumtot1
                                        mCtemkGhm[jmas] = temkGhm/rownumtot1
                                        mCUA[jmas] = UA/rownumtot1
                                        mCNTU[jmas] = NTU/rownumtot1
                                        mCeps[jmas] = eps/rownumtot1
                                        mCRhm[jmas] = Rhm/rownumtot1
                                        mCRcm[jmas] = Rcm/rownumtot1
                                        mCkT[jmas] = kT/rownumtot1
                                        mCQhm[jmas] = Qhm/rownumtot1
                                        mCqhmplot[jmas] = qhmplot/rownumtot1
                                        mCTchm[jmas] = Tchm/rownumtot1
                                        mCTccm[jmas] = Tccm/rownumtot1
                                        mCdeltaTc[jmas] = deltaTc/rownumtot1
                                        mCeta[jmas] = eta/rownumtot1
                                        mCPelem[jmas] = Pelem/rownumtot1
                                        (
                                            Reycm, alphateplfcm, Prcmm, lbdcmm, rocmm, cpcmm, nucmm, wcm, Tcm, temkGcm,
                                            Reyhm, alphateplfhm, Prhmm, lbdhmm, rohmm, cphmm, nuhmm, whm, Thm, temkGhm,
                                            ThmTEGout, UA, NTU, eps, Rhm, Rcm, kT,
                                            Qhm, qhmplot, Tchm, Tccm, deltaTc, eta, Pelem
                                        ) = (
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0,
                                        )
                                        im = 0
                                        imas = 1
                                        jmas += 1
                                    # осреднение параметров для ТЭГ
                                    if jmas > nummax:
                                        for km in range(1, nummax+1):
                                            Reycm += mCReycm[km]
                                            alphateplfcm += mCalphateplfcm[km]
                                            Prcmm += mCPrcmm[km]
                                            lbdcmm += mClbdcmm[km]
                                            rocmm += mCrocmm[km]
                                            cpcmm += mCcpcmm[km]
                                            nucmm += mCnucmm[km]
                                            wcm += mCwcm[km]
                                            Tcm += mCTcm[km]
                                            temkGcm += mCtemkGcm[km]
                                            Reyhm += mCReyhm[km]
                                            alphateplfhm += mCalphateplfhm[km]
                                            Prhmm += mCPrhmm[km]
                                            lbdhmm += mClbdhmm[km]
                                            rohmm += mCrohmm[km]
                                            cphmm += mCcphmm[km]
                                            nuhmm += mCnuhmm[km]
                                            whm += mCwhm[km]
                                            Thm += mCThm[km]
                                            temkGhm += mCtemkGhm[km]
                                            UA += mCUA[km]
                                            NTU += mCNTU[km]
                                            eps += mCeps[km]
                                            Rhm += mCRhm[km]
                                            Rcm += mCRcm[km]
                                            kT += mCkT[km]
                                            Qhm += mCQhm[km]
                                            qhmplot += mCqhmplot[km]
                                            Tchm += mCTchm[km]
                                            Tccm += mCTccm[km]
                                            deltaTc += mCdeltaTc[km]
                                            eta += mCeta[km]
                                            Pelem += mCPelem[km]
                                        Reycm /= nummax
                                        alphateplfcm /= nummax
                                        Prcmm /= nummax
                                        lbdcmm /= nummax
                                        rocmm /= nummax
                                        cpcmm /= nummax
                                        nucmm /= nummax
                                        wcm /= nummax
                                        Tcm /= nummax
                                        temkGcm /= nummax
                                        Reyhm /= nummax
                                        alphateplfhm /= nummax
                                        Prhmm /= nummax
                                        lbdhmm /= nummax
                                        rohmm /= nummax
                                        cphmm /= nummax
                                        nuhmm /= nummax
                                        whm /= nummax
                                        Thm /= nummax
                                        temkGhm /= nummax
                                        UA /= nummax
                                        NTU /= nummax
                                        eps /= nummax
                                        Rhm /= nummax
                                        Rcm /= nummax
                                        kT /= nummax
                                        Qhm /= nummax
                                        qhmplot /= nummax
                                        Tchm /= nummax
                                        Tccm /= nummax
                                        deltaTc /= nummax
                                        eta /= nummax
                                        Pelem /= nummax

                            ThmTEGout = 0
                            for jmas in range(1, nummax+1):
                                ThmTEGout += mThmTEGout[jmas]
                                jm = nummax - jmas
                                if jm == 1:
                                    ThmTEGout1 = ThmTEGout
                            # одинаковое число каналов в группах
                            if cmcannummin == 0:
                                ThmTEGout = ThmTEGout/cmcannummax
                            # число каналов в группах различно
                            else:
                                ThmTEGout = (cmgrmax/cmflownum) * (ThmTEGout/cmcannummax) + (cmgrmin/cmflownum) * (ThmTEGout1/cmcannummin)

                            mNTEG = roNTEG*(
                                2*bsthichm*widthhm*widthcm + 4*0.002*hhm*widthcm + (
                                    deltahm*hhm*(math.trunc(Lhm/shaghm) - 1) + \
                                    deltahm*(Lhm - (math.trunc(Lhm/shaghm) - 1)*deltahm)
                                )*widthcm
                            )
                            mOTEG = roOTEG*(
                                2*bsthiccm*widthhm*widthcm + 4*0.002*hcmm*widthhm + \
                                2*(hTEM - 0.001)*(widthhm*widthcm - (widthhm - 0.04)*(widthcm - 0.04)) + (
                                    deltacm*hcmm*(math.trunc(Lcm/shagcm) - 1) + \
                                    deltacm*(Lcm - (math.trunc(Lcm/shagcm) - 1)*deltacm)
                                )*widthhm
                            )
                            mcmap = rocmm*Arrowcm*widthhm
                            mcmcol = 7800*1.15*(
                                4*widthcm*0.25*((dpatrcms + 0.004)**2 - dpatrcms**2)*math.atan(hcms / (dpatrcms**2 - hcms**2)**0.5) + \
                                6 * 0.002 * 3.14 * 0.25 * (dpatrcms + 0.004)**2 + \
                                ((widthpatrobv + 0.004)*(hcms + 0.004) - widthpatrobv*hcms) * (4*0.03 + 0.5*3.14*rgib*0.5) + \
                                2*cmflownum*widthcm*0.25*((dpatrcmm + 0.004)**2 - dpatrcmm**2)*math.atan(hcmm / (dpatrcmm**2 - hcmm**2)**0.5) + \
                                (2*cmflownum + 1) * 0.002 * 3.14 * 0.25 * (dpatrcmm + 0.004)**2 + \
                                ((widthpatrobv + 0.004)*(hcmm + 0.004) - widthpatrobv*hcmm) * (
                                    2*(nNTEG - 1)*0.03 + \
                                    (cmcannummax - 2 + (cmgrmax - 1)*(cmcannummax - 1) + cmgrmin*(cmcannummin - 1))*3.14*rgib
                                ) + \
                                3.14*0.25*((dpatrcms + 0.004)**2 - dpatrcms**2) + \
                                (2*cmflownum - 1)*0.5*3.14*0.25*((dpatrcmm + 0.004)**2 - dpatrcmm**2) + \
                                ntrsoed * (heightsum + 3.14*3*dtrsoed) * 3.14 * 0.25 * ((dtrsoed + 0.004)**2 - dtrsoed**2)
                            )
                            mcm = mcmap + rocmm*(
                                4 * widthcm * 3.14 * 0.25 * dpatrcms**2 + widthpatrobv * hcms * (4*0.03 + 0.5*3.14*rgib*0.5) + \
                                2 * cmflownum * widthcm * 3.14 * 0.25 * dpatrcmm**2 + widthpatrobv*hcmm*(
                                    2*(nNTEG - 1)*0.03 + (
                                        cmcannummax - 2 + \
                                        (cmgrmax - 1)*(cmcannummax - 1) + cmgrmin*(cmcannummin - 1)
                                    )*3.14*rgib
                                ) + \
                                3.14 * 0.25 * dpatrcms**2 + \
                                (2*cmflownum - 1) * 0.5 * 3.14 * 0.25 * dpatrcmm**2 + \
                                ntrsoed * (heightsum + 3.14*3*dtrsoed) * 3.14 * 0.25 * dtrsoed**2
                            )

                            mhmap = rohmm*Arrowhm*widthhm
                            acol = acol2 if acol2>acol3 else acol3
                            hcol = (acol - dpatrhm)/0.7279
                            mhmcol = 2*7800*(
                                (fldiam/2) * (heightsum*widthhm - acol1*acol2) + \
                                dhidrhm * ((acol1 + 0.004)*(acol2 + 0.004) - acol1*acol2) + \
                                0.333*hobt*(
                                    (
                                        (acol2 + 0.004)*(acol3 + 0.004) + \
                                        ((acol2 + 0.004)*(acol3 + 0.004)*(acol1 + 0.004)*(acol2 + 0.004))**0.5 + \
                                        (acol1 + 0.004)*(acol2 + 0.004)
                                    ) - \
                                    (acol2*acol3 + (acol2*acol3*acol1*acol2)**0.5 + acol1 * acol2)
                                ) + \
                                dpatrhm*0.5*((acol2 + 0.004)*(acol3 + 0.004) - acol2*acol3) + \
                                0.333*hcol*(
                                    (
                                        (acol2 + 0.004)*(acol3 + 0.004) + \
                                        ((acol2 + 0.004) * (acol3 + 0.004) * (dpatrhm + 0.004)**2)**0.5 + \
                                        (dpatrhm + 0.004)**2
                                    ) - (acol2*acol3 + (acol2 * acol3 * dpatrhm**2)**0.5 + dpatrhm**2)
                                ) + \
                                dpatrhm * 0.5 * 3.14 * 0.25 * ((dpatrhm + 0.004)**2 - dpatrhm**2) + \
                                (nNTEG - 1)*acol2*(2*0.002*(hTEM + 0.003) + \
                                2*((fldiam/2) + dhidrhm)*0.001 + 0.5*hobt*(HOTEGm + 0.004) - 0.5 * (HOTEGm + 0.002)**2 / 0.5359)
                            )
                            mhm = mhmap +2*rohmm*(
                                (fldiam/2)*acol1*acol2 + dhidrhm*acol1*acol2 + \
                                0.333 * hobt * (acol2*acol3 + (acol2*acol3*acol1*acol2)**0.5 + acol1*acol2) + \
                                dpatrhm*0.5*acol2*acol3 + \
                                0.333 * hcol * (acol2*acol3 + (acol2 * acol3 * dpatrhm**2)**0.5 + dpatrhm**2) + \
                                dpatrhm * 0.5 * 3.14 * 0.25 * dpatrhm**2 - \
                                (nNTEG - 1) * acol2 * ((HOTEGm + 0.004) * ((fldiam/2) + dhidrhm + 0.002) + 0.5*hobt*(HOTEGm + 0.004))
                            )

                            mgask2 = 2300 * 0.002 * (heightsum*widthhm - acol1*acol2)
                            LTEG = widthcm + 2*dpatrhm + 2*hcol + 2*hobt + 2*dhidrhm + fldiam + 0.004

                            mgask1 = 2300 * 0.002 * (widthhm*widthcm - ((widthhm - 0.04)*(widthcm - 0.04)))
                            mTES = mTEM*TEMlayq2
                            mTEG = nNTEG*(mNTEG + mOTEG) + 2*nNTEG*(mTES + mgask1) + 2*mgask2 + mcmcol + mhmcol + mcm + mhm
                            BTEG = widthhm + 1.15*(0.06 + dpatrcmm + 0.004 + dpatrcms + 0.004 + 5.5*dtrsoed + 0.002)

                            # results_dict['message'] = 'Расчёт выполнен'
                            # Данные в массивах
                            results_dict['TEM_distrib_TEG'] = TEM_distrib_TEG
                            results_dict['Tcm'] = np.array(params_dict['Tcm']).reshape(nummax, rownumtot1, nTEMrowhm)
                            results_dict['Thm'] = np.array(params_dict['Thm']).reshape(nummax, rownumtot1, nTEMrowhm)
                            results_dict['Pelem_arr'] = np.array(params_dict['Pelem']).reshape(nummax, rownumtot1, nTEMrowhm)
                            results_dict['Qhm_arr'] = np.array(params_dict['Qhm']).reshape(nummax, rownumtot1, nTEMrowhm)
                            results_dict['Rhm_arr'] = np.array(params_dict['Rhm']).reshape(nummax, rownumtot1, nTEMrowhm)
                            results_dict['Rcm_arr'] = np.array(params_dict['Rcm']).reshape(nummax, rownumtot1, nTEMrowhm)
                            results_dict['Tchm_arr'] = np.array(params_dict['Tchm']).reshape(nummax, rownumtot1, nTEMrowhm)
                            results_dict['Tccm_arr'] = np.array(params_dict['Tccm']).reshape(nummax, rownumtot1, nTEMrowhm)
                            # Массогабаритные характеристики
                            results_dict['widthhm'] = widthhm
                            results_dict['BTEG'] = BTEG
                            results_dict['widthcm'] = widthcm
                            results_dict['LTEG'] = LTEG
                            results_dict['heightsum'] = heightsum
                            results_dict['mTEG'] = mTEG
                            results_dict['mTEG_empty'] = mTEG - mcm - mhm
                            results_dict['nNTEG'] = nNTEG
                            results_dict['cmflownum'] = cmflownum
                            results_dict['cmcannummax'] = cmcannummax
                            results_dict['cmcannummin'] = cmcannummin
                            results_dict['cmgrmax'] = cmgrmax
                            results_dict['TEMlayq2'] = TEMlayq2
                            results_dict['rownumtot1'] = rownumtot1
                            results_dict['nTEMrowhm'] = nTEMrowhm
                            # Параметры теплопередачи и электрогенерации
                            results_dict['UA'] = UA
                            results_dict['NTU'] = NTU
                            results_dict['eps'] = eps
                            results_dict['Rhm'] = Rhm
                            results_dict['Rcm'] = Rcm
                            results_dict['kT'] = kT
                            results_dict['Qhm'] = Qhm
                            results_dict['qhmplot'] = qhmplot
                            results_dict['Tchm'] = Tchm
                            results_dict['Tccm'] = Tccm
                            results_dict['deltaTc'] = deltaTc
                            results_dict['eta'] = eta
                            results_dict['Pelem'] = Pelem
                            results_dict['PTEG'] = PTEG
                            results_dict['Nrtccm'] = Nrtccm
                            # results_dict['Pus'] = Pus
                            # Параметры НТЭГ
                            results_dict['Arthm_HE'] = Arthm/nNTEG
                            results_dict['Arthm'] = Arthm
                            results_dict['massflow_hm_HE'] = Arrowhm*whm*rohmm/nNTEG
                            results_dict['massflow_hm'] = hmflownum*Arrowhm*whm*rohmm/nNTEG
                            results_dict['rtchmap'] = rtchmap
                            results_dict['rtchm'] = rtchm
                            results_dict['whm'] = whm
                            results_dict['ThmTEGout'] = ThmTEGout
                            results_dict['temkGhm'] = temkGhm
                            results_dict['alphateplfhm'] = alphateplfhm
                            # Параметры ОТЭГ
                            results_dict['Artcm_HE'] = Artcm/nNTEG
                            results_dict['Artcm'] = Artcm
                            results_dict['massflow_cm_HE'] = Arrowcm*wcm*rocmm/nNTEG
                            results_dict['massflow_cm'] = cmflownum*Arrowcm*wcm*rocmm/nNTEG
                            results_dict['rtccmap'] = rtccmap
                            results_dict['rtccm'] = rtccm
                            results_dict['wcm'] = wcm
                            results_dict['TcmTEGout'] = TcmTEGout
                            results_dict['temkGcm'] = temkGcm
                            results_dict['alphateplfcm'] = alphateplfcm

                    else:
                        results_dict['message'] = 'Расчёт невозможен. Превышено сопротивление по холодной среде.'

                else:
                    results_dict['message'] = 'Расчёт невозможен. Несоответствие габаритов ТЭМ и ТЭГ.'

            else:
                results_dict['message'] = 'Расчёт невозможен. Превышено сопротивление по горячей среде.'

        else:
            results_dict['message'] = 'Расчёт невозможен. Не удалось подобрать фланцевые бобышки для газового тракта.'

    else:
        results_dict['message'] = 'Расчёт невозможен. Не удалось подобрать стяжной крепёж.'

    return results_dict







