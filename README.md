### 6 key api of pytorch
- model = Model(arg_model) # 1. model
- opt = Optimizer(arg_opt, model.parameters()) # 2. optimizer
- y = model(x) # 3. forward
- loss = loss_f(y, t) # 4. criterion
- loss.backward() # 5. backward
- opt.step() # 6. update

### model training
```mermaid
graph LR
    init --> |init_model| model
    init --> |init_opt| opt
    model --> model.parameters
    opt --> model.parameters
    model --> |forward| y
    y --> |criteria| loss
    loss --> |backward| opt
    opt --> |update| model.parameters
```

### files structure
```mermaid
graph TD
    dataloader.py --> config.py
    model.py --> config.py
    model.py --> optimizer.py
    optimizer.py --> config.py
    criterion.py --> config.py
    config.py --> train.py
    config.py --> val.py
    config.py --> test.py
    train.py --> run.py
    val.py --> run.py
    test.py --> run.py
```