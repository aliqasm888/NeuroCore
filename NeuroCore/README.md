# qasm - Mini Neural Network Library


تم تنفيذ مكتبة مصغرة للشبكات العصبونية (Mini Neural Network)
python من الصفر باستخدام لغة 
NumPy ومكتبة

- (Forward Propagation)
-   (Backpropagation)
- حساب التدرجات وتحديث الأوزان
- كيفية استخدام خوارزميات تحديث مختلفة (Optimizers)
- كيفية البحث عن أفضل قيم للـ hyperparameters
- بناء شبكات عصبونية  (robust neural networks)

## البنية العامة للمكتبة

المكتبة مبنية على الصفوف الأساسية

### 1. الصف الأساسي: `Layer` (Base Class)
- يحتوي على دالتين أساسيتين: `forward(x)` و `backward(dout)`

### 2. طبقة الاتصال الكامل: `Affine` (Dense)
- تقوم بعملية الضرب المصفوفي: `input @ weight + bias`
- تحسب التدرجات `dW` و `db`

### 3. طبقات التنشيط (Activation Functions)
تم تنفيذ:
- `Linear`
- `ReLU`
- `Sigmoid`
- `Tanh`

كل واحدة تنفذ `forward` و `backward` بشكل element-wise.

### 4. توابع الخسارة (Loss Functions)
تم تنفيذ:
- `MeanSquaredError`
- `SoftmaxCrossEntropy`

### 5. الشبكة العصبونية: `NeuralNetwork`
- الصف الرئيسي لبناء الشبكة.
- يدعم `add_layer`، `set_loss`، `predict`، `accuracy`، وحساب التدرجات.



- **Trainer**: لتنظيم عملية التدريب 
- **Optimizers**: تم تنفيذ خوارزميات التحديث  (SGD، Momentum، AdaGrad، Adam) 
- **Hyperparameter Tuning**: Grid Search بسيط لتجربة قيم مختلفة للـ hyperparameters (learning rate، batch size، optimizer type، hidden size، activation function)hyperparameter".


