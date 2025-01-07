# 导入库
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
# 加载数据集
#导入数据
data = pd.read_excel('tree_cart.xlsx')
# 将分类特征编码为数值
data['喉咙痛'] = data['喉咙痛'].map({'是': 1, '否': 0})
data['咳嗽'] = data['咳嗽'].map({'是': 1, '否': 0})
data['体温'] = data['体温'].map({'很高': 2, '高': 1, '正常': 0})
data['是否感冒'] = data['是否感冒'].map({'是': 1, '否': 0})
#划分分类特征和目标变量
X = data.iloc[:,:-1].values#返回2维数组，返回除了最后一列的所有列
y = data['是否感冒'].values
# 创建决策树分类器实例
dtree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
# 训练决策树分类器
dtree.fit(X, y)
# 使用StringIO来捕获决策树的dot语言描述
dot_data = StringIO()
# 导出决策树为dot格式
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                class_names = ['否','是'],feature_names = ['喉咙痛','咳嗽','体温'])
#解决中文乱码的问题
dot_data_val = dot_data.getvalue()
dot_data_val = dot_data_val.replace('helvetica', 'SimSun')
# 使用pydotplus将dot语言转换为图形
graph = pydotplus.graph_from_dot_data(dot_data_val)
# 将决策树图形保存为PNG文件
graph.write_png('tree.png')



