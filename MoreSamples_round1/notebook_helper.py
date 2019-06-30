"""Small functions to declutter notebook"""
import arviz as az
import numpy as np

def check_dataset_true(dataset):
    return all(var.min() for var in dataset.variables.values())
    
def check_limits(inference_data, ess=500, rhat=1.01, mcse=None):
    """Create each variable against specified limit."""
    
    funcs = (
        lambda x: az.ess(x, method="bulk"),
        lambda x: az.ess(x, method="tail"),
        az.rhat,
        lambda x: az.mcse(x, method="mean"), 
        lambda x: az.mcse(x, method="sd"),
        lambda x: az.mcse(x, method="quantile", prob=0.5),
    )
    
    limits = ((ess, "gt"), 
              (ess, "gt"), 
              (rhat, "lt"), 
              (mcse, "lt"), 
              (mcse, "lt"), 
              (mcse, "lt"),
             )
    
    condition = True
    for func, (limit, rule) in zip(funcs, limits):
        if limit is None:
            continue
        if rule == "gt":
            condition &= check_dataset_true(func(inference_data).min() > limit)
        else:
            condition &= check_dataset_true(func(inference_data).max() < limit)
        if not condition:
            return False
    return True
    
def create_plot(idata, ax, func, method=None, variables=None, limit=None, legend=True, rule="min", name=None):
    
    if method is None:
        data = func(idata)
    else:
        data = func(idata, method=method)
    
    n = idata.posterior.dims["draw"] * idata.posterior.dims["chain"]
    
    lines = {}
    if variables is not None:
        for var in variables: # ["alpha", "rho", "sigma"]
            data_ = data[var]
            if data_.shape:
                for i, value in np.ndenumerate(data_):
                    line, = ax.plot([n], [float(value)], marker=".", label=f"{var}_{str(i)}" if legend else "_nolegend_")
                    lines[(var, i)] = line

            else:
                line, = ax.plot([n], [float(data_)], marker=".", label=var if legend else "_nolegend_")
                lines[var] = line
    
    if rule == "min":
        minimum_data = float(min(data.min().variables.values()))
        line, = ax.plot([n], minimum_data, marker='o', label="minimum" if legend else "_nolegend_")
        lines["__minimum"] = line
    elif rule == "max":
        maximum_data = float(max(data.max().variables.values()))
        line, = ax.plot([n], maximum_data, marker='o', label="maximum" if legend else "_nolegend_")
        lines["__maximum"] = line
    
    if limit is not None:
        line = ax.axhline(limit, lw=2, color="k", label="target" if legend else "_nolegend_")
        lines["__plot_limit"] = line
    
    if legend:
        cols = len(lines) // 3 + 1
        ax.legend(loc=(1.01, 0.1), ncol=cols)
        
    if name is not None:
        ax.set_ylabel(name)
    
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(0, n*1.1)
    return ax, lines


def update_plot(idata, ax, func, method=None, lines=None, variables=None):
    
    if method is None:
        data = func(idata)
    else:
        data = func(idata, method=method)
    n = idata.posterior.dims["draw"] * idata.posterior.dims["chain"]
    
    max_y = 0
    if lines is None:
        lines = {}
    if variables is not None:
        for var in variables:
            data_ = data[var]
            if data_.shape:
                for i, value in np.ndenumerate(data_):
                    if (var, i) in lines:
                        line = lines[(var, i)]
                        line.set_xdata(np.append(line.get_xdata(), n))
                        line.set_ydata(np.append(line.get_ydata(), value))
                        if max(line.get_ydata()) > max_y:
                            max_y = max(line.get_ydata())

            else:
                if var in lines:
                    line = lines[var]
                    line.set_xdata(np.append(line.get_xdata(), n))
                    line.set_ydata(np.append(line.get_ydata(), float(data_)))
                    if max(line.get_ydata()) > max_y:
                            max_y = max(line.get_ydata())
    
    if "__minimum" in lines:
        minimum_data = float(min(data.min().variables.values()))
        line = lines["__minimum"]
        line.set_xdata(np.append(line.get_xdata(), n))
        line.set_ydata(np.append(line.get_ydata(), minimum_data))
        if max(line.get_ydata()) > max_y:
            max_y = max(line.get_ydata())
    elif "__maximum" in lines:
        maximum_data = float(max(data.max().variables.values()))
        line = lines["__maximum"]
        line.set_xdata(np.append(line.get_xdata(), n))
        line.set_ydata(np.append(line.get_ydata(), maximum_data))
        if max(line.get_ydata()) > max_y:
            max_y = max(line.get_ydata())
        
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(0, n*1.1)
    return ax, lines