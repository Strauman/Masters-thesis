import numpy as np
import matplotlib
matplotlib.use("PDF")
from matplotlib import pyplot as plt
from .export_help import take_idc

def filter_area_quantiles(arrs, idx=0, n_out=-1, pool_size=None, num_quantiles=10, lower_tail=True):
    arrs = ensure_lstcp(arrs)
    areas = np.argsort(np.sum(arrs[idx], axis=(1, 2)))[::-1]
    # plt.hist(areas, bins='auto')
    # saveas()
    quantiles = np.linspace(0, 1, num_quantiles)
    q_thresh = np.quantile(areas, quantiles)
    # print(f"Quantile val: {q_thresh}. a_max={np.max(areas)}, a_min={np.min(areas)}")
    arr_list = []
    for i, thr in enumerate(q_thresh):
        if lower_tail:
            area_idx = areas[areas <= thr]
        else:
            area_idx = areas[areas >= thr]
        x = quantiles[i]
        y = take_idc(arrs[2], area_idx, n_out=-1, pool_size=None)
        arr_list.append((x, y))
    return arr_list

def f1_for_a_quantiles():
    tr = tm_trainer
    handle = DS_t.tf_it_handle
    sess = tf.get_default_session()
    print(f"Restore for area conditioned grid for {tr.name}")
    ask_restore()
    # Select data with good F1_scores after ADDA
    tr.init_it_val()
    X, Y, F1 = run_epoch(tr.feeds_val(), tr.inputs, tr.labels, tr.indiv_f1)
    F1_all = np.mean(F1)
    Qarrs = filter_area_quantiles([X, Y, F1], num_quantiles=10, lower_tail=False)
    F1_score_lists = []
    # new_handle=export_help.handle_from_data(x,y)
    # F1_scores_indiv=run_epoch({handle: new_handle}, tr.indiv_f1)
    for q, f1 in Qarrs:
        F1_score_lists.append((q, np.mean(f1)))
    F1_score_lists = list(zip(*F1_score_lists))
    best_thr_idx = np.argmax(F1_score_lists[1])
    best_thr, best_val = [f[best_thr_idx] for f in F1_score_lists]
    print(f"Best f1: {best_val} @ thr {best_thr}")
    plt.plot(F1_score_lists[0], F1_score_lists[1])
    saveas()
    

def domain_grid_fig(tr, ds):
    sess = tf.get_default_session()
    tr.init_it_val()
    f1_best, f1_thresh = tr.best_f1(concise=True, cheat=False)
    print("Best f1:", f1_best)
    tr.output_threshold = f1_thresh
    tr.init_it_val()
    X_f, Y_f, F1 = run_epoch(tr.feeds_val(), *arr_idc.tf_vars())
    sample_from = 10
    best_idx = np.argsort(F1)[::-1]
    # best_idx=np.arange(F1.shape[0])
    # np.random.shuffle(best_idx)
    n = 5
    chosen = np.random.choice(sample_from, n)
    X_f = X_f[best_idx][:sample_from][chosen]
    Y_f = Y_f[best_idx][:sample_from][chosen]
    F1 = F1[best_idx][:sample_from][chosen]
    custom_data_handle = export_help.handle_from_data(X_f, Y_f)
    YhatBest = run_epoch({ds.tf_it_handle: custom_data_handle}, tr.outputs)
    nr = n
    nc = 3
    fig = plt.figure()
    gs = gridspec.GridSpec(nr, nc)

    def label_y(r, c):
        if c == 0:
            export_help.get_ax(r, c, gs).set_ylabel(f"F1:{np.array2string(np.array(F1[r]))}")

    export_help.export_domain_grid(X_f, Y_f, YhatBest, gs, label_y)

    # sess.run(new_iterator.initializer)
    # print(np.all(np.equal(X_redone, X_redo)))
    # for x_f,x_r in zip(X_f, X_redone):

def f1_large_areas():
    tr = tm_trainer
    sess=tf.get_default_session()
    print(f"Restore for area conditioned grid for {tr.name}")
    ask_restore()
    # Select data with good F1_scores after ADDA
    tr.init_it_val()
    X, Y, F1 = run_epoch(tr.feeds_val(), *arr_idc.tf_arrs(tr))
    F1_pre=np.mean(F1)
    nx_pre=X.shape[0]
    print(f"Input starts: {nx_pre}")
    handle = DS_t.tf_it_handle
    X,Y,F1=filter_area_quantile([X,Y,F1], 1, quantile=0.5)
    nx_post=X.shape[0]
    print(f"Xs left: {nx_post}. Ratio {nx_post/nx_pre}")
    new_handle=export_help.handle_from_data(X, Y)
    F1_scores_indiv=run_epoch({handle: new_handle}, tr.indiv_f1)
    print(f"Area scores: idiv_mean={F1_pre}->{np.mean(F1_scores_indiv)}")
    return F1_scores_indiv
