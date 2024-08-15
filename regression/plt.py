import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import gaussian_kde


def plt_contour(y_true, y_pred):

    # 为了避免过度绘图，这里随机选择10000个样本绘制
    indices = np.random.choice(range(len(y_true)), size=10000, replace=False)
    sampled_y_true = y_true[indices]
    sampled_y_pred = y_pred[indices]

    # 创建画布
    plt.figure(figsize=(8, 6))

    # 使用直方图的二维表示形式来获取x和y的直方图值
    h, xedges, yedges, image = plt.hist2d(sampled_y_pred, sampled_y_true, bins=50, density=True, cmap='viridis')

    # 绘制等高线图
    contours = plt.contour(xedges[:-1], yedges[:-1], h, 5, colors='white')

    # 等高线标签
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    # 添加对角线，表示完美预测的位置
    plt.plot([sampled_y_true.min(), sampled_y_true.max()], [sampled_y_true.min(), sampled_y_true.max()], 'k--', lw=2)

    # 坐标轴标签
    plt.xlabel('Predicted Ejection Fraction')
    plt.ylabel('Actual Ejection Fraction')

    # 图标题
    plt.title("Contour map of predicted vs. actual ejection fraction (sampled)")

    # 显示颜色栏
    plt.colorbar()

    # 显示绘图
    plt.show()



def plot_contour(true_ef, predicted_ef):

    num_samples = 10000
    indices = np.random.choice(len(true_ef), num_samples, replace=False)
    predicted_ef_subset = predicted_ef[indices]
    true_ef_subset = true_ef[indices]

    # 创建网格数据
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # 计算等高线的值
    for i in range(len(true_ef_subset)):
        ix = np.argmin(np.abs(x - true_ef_subset[i]))
        iy = np.argmin(np.abs(y - predicted_ef_subset[i]))
        Z[iy, ix] += 1

    # 绘制等高线图
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=np.arange(1, np.max(Z)+1), cmap='coolwarm')
    plt.clabel(contour, inline=True, fontsize=10)
    plt.xlabel('True Ejection Fraction')
    plt.ylabel('Predicted Ejection Fraction')
    plt.title('Contour Plot of Ejection Fraction Prediction')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()



def plot_sns_contour(true_ef, predicted_ef):

    # 从测试集中随机选择10000个样本
    num_samples = 10000
    indices = np.random.choice(len(true_ef), num_samples, replace=False)
    predicted_ef_subset = predicted_ef[indices]
    true_ef_subset = true_ef[indices]

    # 使用Seaborn的jointplot来绘制散点图和等高线图
    g = sns.jointplot(x=true_ef_subset, y=predicted_ef_subset, kind="kde", space=0, color="g", xlim=(20, 80), ylim=(20, 80))

    # 绘制对角线，表示完美预测的线
    g.ax_joint.plot([20, 80], [20, 80], linestyle="--", color="black")

    # 设置坐标轴标签和标题
    g.set_axis_labels('True Ejection Fraction', 'Predicted Ejection Fraction')
    g.fig.suptitle('Contour Plot of Ejection Fraction Prediction')

    # 调整子图间距
    plt.subplots_adjust(top=0.9)

    # 显示图形
    plt.show()



def plot_scatter_alpha(y_true, y_pred):

    # plt.figure(figsize=(10, 8))
    # plt.scatter(sample_y_true, sample_y_pred, alpha=0.1, s=10, c='blue', cmap='viridis')  # alpha值可以根据需要调整
    plt.scatter(y_true, y_pred, alpha=0.4, s=25, c='orange')
    plt.plot([0, 100], [0, 100], 'k--')  # 绘制y=x的参考线
    # plt.grid(True)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Prediction vs. True Value')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    # plt.colorbar(label='Density')
    plt.show()

def plot_scatter_cmap(y_true, y_pred):

    pearson_coeff, _ = stats.pearsonr(y_true, y_pred)
    print("Pearson correlation coefficient: ",  pearson_coeff)
    # 假设y_true是测试集的真实值，y_pred是模型预测的值
    indices = np.random.choice(range(len(y_true)), 10000, replace=False)
    sample_y_true = y_true[indices]
    sample_y_pred = y_pred[indices]

    # 计算点的密度
    values = np.vstack([sample_y_true, sample_y_pred])
    kernel = gaussian_kde(values)(values)

    # 排序点，密度高的点会被后绘制，这样在图上更突出
    idx = kernel.argsort()
    sample_y_true, sample_y_pred, kernel = sample_y_true[idx], sample_y_pred[idx], kernel[idx]

    fig, ax = plt.subplots()
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=25, edgecolor='none', cmap='Oranges', alpha=0.6)
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=50, edgecolor='none', cmap='inferno', alpha=0.7)
    scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=50, edgecolor='none', cmap='magma', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Density')

    # 添加辅助线
    ax.plot([0, 100], [0, 100], 'w--', linewidth=2)  # 白色虚线为y=x
    error_std = np.std(y_pred - y_true)
    # error_std = np.std(sample_y_pred - sample_y_true)
    ax.plot([0, 100], [0+error_std, 100+error_std], 'w:', linewidth=1.5)  # 上界误差线
    ax.plot([0, 100], [0-error_std, 100-error_std], 'w:', linewidth=1.5)  # 下界误差线

    # 设置坐标轴和背景
    ax.set_xlabel('True LVEF (%)')
    ax.set_ylabel('Predicted LVEF (%)')
    fig.subplots_adjust(top=0.9)
    ax.set_title('Scatter plot on the testing set')
    ax.set_facecolor('lightgray')  # 设置浅灰色背景
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')  # 添加网格线增强可读性
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 添加MAE注释
    mae_annotation = "MAE: 5.28 (5.23 - 5.33)"
    mae_color = "royalblue"
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=mae_annotation,
                                markerfacecolor=mae_color, markersize=10, markeredgewidth=0)]
    
    ax.legend(handles=legend_elements, loc='lower right')

    plt.show()



def plot_scatter_test(y_true, y_pred):

    pearson_coeff, _ = stats.pearsonr(y_true, y_pred)
    print("Pearson correlation coefficient: ",  pearson_coeff)

    # 计算点的密度
    values = np.vstack([y_true, y_pred])
    kernel = gaussian_kde(values)(values)

    # 排序点，密度高的点会被后绘制，这样在图上更突出
    idx = kernel.argsort()
    y_true, y_pred, kernel = y_true[idx], y_pred[idx], kernel[idx]

    fig, ax = plt.subplots()
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=25, edgecolor='none', cmap='Oranges', alpha=0.6)
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=50, edgecolor='none', cmap='inferno', alpha=0.7)
    scatter = ax.scatter(y_true, y_pred, c=kernel, s=50, edgecolor='none', cmap='magma', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Density')

    # 添加辅助线
    ax.plot([0, 100], [0, 100], 'w--', linewidth=2)  # 白色虚线为y=x
    error_std = np.std(y_pred - y_true)
    # error_std = np.std(sample_y_pred - sample_y_true)
    ax.plot([0, 100], [0+error_std, 100+error_std], 'w:', linewidth=1.5)  # 上界误差线
    ax.plot([0, 100], [0-error_std, 100-error_std], 'w:', linewidth=1.5)  # 下界误差线

    # 设置坐标轴和背景
    ax.set_xlabel('True LVEF (%)')
    ax.set_ylabel('Predicted LVEF (%)')
    fig.subplots_adjust(top=0.9)
    ax.set_title('Scatter plot on the testing set')
    ax.set_facecolor('lightgray')  # 设置浅灰色背景
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')  # 添加网格线增强可读性
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)


    plt.show()


def plot_scatter_sub(y_true, y_pred):

    pearson_coeff, _ = stats.pearsonr(y_true, y_pred)
    print("Pearson correlation coefficient: ",  pearson_coeff)

    # 计算点的密度
    values = np.vstack([y_true, y_pred])
    kernel = gaussian_kde(values)(values)

    # 排序点，密度高的点会被后绘制，这样在图上更突出
    idx = kernel.argsort()
    y_true, y_pred, kernel = y_true[idx], y_pred[idx], kernel[idx]

    # 使用 Passing–Bablok 回归
    # 添加常数项，构建模型
    X = sm.add_constant(y_true)
    model = sm.OLS(y_pred, X).fit()
    predicted = model.predict(X)

    # 获取回归系数和置信区间
    slope, intercept = model.params[1], model.params[0]
    conf_int = model.conf_int()
    ci_slope, ci_intercept = conf_int[1], conf_int[0]

    fig, ax = plt.subplots()
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=25, edgecolor='none', cmap='Oranges', alpha=0.6)
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=50, edgecolor='none', cmap='inferno', alpha=0.7)
    scatter = ax.scatter(y_true, y_pred, c=kernel, s=50, edgecolor='none', cmap='magma', alpha=0.7)
    # scatter = ax.scatter(y_true, y_pred, c=kernel, s=50, edgecolor='none', cmap='inferno', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Density')

    # 调整坐标轴范围
    x_range = (10, 45)
    y_range = (20, 55)
    # x_range = (25, 60)
    # y_range = (10, 90)
    # x_range = (0, 100)
    # y_range = (0, 100)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)


    x_full_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500)

    ax.plot(x_full_range, x_full_range, 'darkblue', linestyle='--', linewidth=1, label='Perfect Concordance')

    # 设置坐标轴和背景
    ax.set_xlabel('True LVEF (%)')
    ax.set_ylabel('Predicted LVEF (%)')
    # ax.set_title('Scatter plot for subgroup of LVEF≤35%')
    ax.set_facecolor('lightgray')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')

    plt.subplots_adjust(top=0.9)

    # 添加MAE注释
    mae_annotation = "MAE: 6.67"
    mae_color = "royalblue"
    legend_elements = [plt.Line2D([0], [0], color='darkblue', lw=1, linestyle='--', label='Perfect Concordance'),
                    plt.Line2D([0], [0], marker='o', color='w', label=mae_annotation,
                                markerfacecolor=mae_color, markersize=10, markeredgewidth=0)]
    
    ax.legend(handles=legend_elements, loc='lower right')

    plt.show()



def plot_scatter_sub3(y_true, y_pred):

    pearson_coeff, _ = stats.pearsonr(y_true, y_pred)
    print("Pearson correlation coefficient: ",  pearson_coeff)
    # 假设y_true是测试集的真实值，y_pred是模型预测的值
    indices = np.random.choice(range(len(y_true)), 2000, replace=False)
    sample_y_true = y_true[indices]
    sample_y_pred = y_pred[indices]

    # 计算点的密度
    values = np.vstack([sample_y_true, sample_y_pred])
    kernel = gaussian_kde(values)(values)

    # 排序点，密度高的点会被后绘制，这样在图上更突出
    idx = kernel.argsort()
    sample_y_true, sample_y_pred, kernel = sample_y_true[idx], sample_y_pred[idx], kernel[idx]

    # 使用 Passing–Bablok 回归
    # 添加常数项，构建模型
    X = sm.add_constant(sample_y_true)
    model = sm.OLS(sample_y_pred, X).fit()
    predicted = model.predict(X)

    # 获取回归系数和置信区间
    slope, intercept = model.params[1], model.params[0]
    conf_int = model.conf_int()
    ci_slope, ci_intercept = conf_int[1], conf_int[0]

    fig, ax = plt.subplots()
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=25, edgecolor='none', cmap='Oranges', alpha=0.6)
    # scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=50, edgecolor='none', cmap='inferno', alpha=0.7)
    scatter = ax.scatter(sample_y_true, sample_y_pred, c=kernel, s=50, edgecolor='none', cmap='magma', alpha=0.7)
    # scatter = ax.scatter(y_true, y_pred, c=kernel, s=50, edgecolor='none', cmap='inferno', alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Density')

    # 调整坐标轴范围
    x_range = (40, 90)
    y_range = (10, 100)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    x_full_range = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 500)
    
    ax.plot(x_full_range, x_full_range, 'darkblue', linestyle='--', linewidth=1, label='Perfect Concordance')

    # 设置坐标轴和背景
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    # ax.set_title('Scatter Plot for subgroup')
    ax.set_facecolor('lightgray')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')

    plt.subplots_adjust(top=0.9)

    # 添加MAE注释
    mae_annotation = "MAE: 5.22"
    mae_color = "royalblue"
    legend_elements = [plt.Line2D([0], [0], color='darkblue', lw=1, linestyle='--', label='Perfect Concordance'),
                    plt.Line2D([0], [0], marker='o', color='w', label=mae_annotation,
                                markerfacecolor=mae_color, markersize=10, markeredgewidth=0)]
    
    ax.legend(handles=legend_elements, loc='lower right')

    plt.show()



def bootstrap_metric(arg1, arg2, fun, num_samples=10000):
    results = []
    arg1, arg2 = np.array(arg1), np.array(arg2)

    for _ in range(num_samples):
        index = np.random.choice(len(arg1), len(arg1))
        results.append(fun(arg1[index], arg2[index]))

    results = sorted(results)
    percentile_05 = results[round(0.05 * len(results))]
    percentile_95 = results[round(0.95 * len(results))]

    return fun(arg1, arg2), percentile_05, percentile_95


def concordance_correlation_coefficient(y_true, y_pred):
    """计算一致相关性系数 (Concordance Correlation Coefficient, CCC)"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.cov(y_true, y_pred)[0, 1]
    pearson_corr = covariance / (np.std(y_true) * np.std(y_pred))
    ccc = (2 * pearson_corr * np.std(y_true) * np.std(y_pred)) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc


def pearsonr_only_corr(x, y):
    corr, _ = stats.pearsonr(x, y)
    return corr


def plot_altman(y_true, y_pred):
    # 假设y_true是测试集的真实值，y_pred是模型预测的值
    indices = np.random.choice(range(len(y_true)), 10000, replace=False)
    sample_y_true = y_true[indices]
    sample_y_pred = y_pred[indices]

    pearson_coeff, _ = stats.pearsonr(sample_y_true, sample_y_pred)
    print("Pearson correlation coefficient: ",  pearson_coeff)

    pear_r = bootstrap_metric(sample_y_true, sample_y_pred, pearsonr_only_corr)
    print(pear_r)

    # 计算一致相关性系数
    # ccc = concordance_correlation_coefficient(sample_y_true, sample_y_pred)
    # print(f"Concordance Correlation Coefficient (CCC) for 10000 samples: {ccc}")
    # ccc = bootstrap_metric(sample_y_true, sample_y_pred, concordance_correlation_coefficient)
    # print(ccc)

    means = (sample_y_true + sample_y_pred) / 2
    diffs = sample_y_pred - sample_y_true
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    # plt.figure(figsize=(10, 5))
    plt.scatter(means, diffs, color='blue', alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle=':')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle=':')
    plt.title('Bland-Altman Plot')
    plt.xlabel('Means of True and Predicted Values')
    plt.ylabel('Differences between Predicted and True Values')
    plt.show()



def plot_scatter_kde(y_true, y_pred):
    sns.kdeplot(x=y_true, y=y_pred, cmap="Blues", fill=True, bw_adjust=.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')  # 绘制y=x的参考线
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('2D Density Plot of Prediction vs. True Value')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()


def plot_hexbin(y_true, y_pred):
    plt.hexbin(y_true, y_pred, gridsize=50, cmap='Oranges', edgecolors='none')
    plt.colorbar()  # 显示颜色条
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Hexbin Plot of Prediction vs True Value')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()