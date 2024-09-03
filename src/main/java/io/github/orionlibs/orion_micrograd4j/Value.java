package io.github.orionlibs.orion_micrograd4j;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * stores a single scalar value and its gradient
 */
public class Value
{
    private float data;
    private String operation;
    private float grad;
    private Set<Value> previous;
    private BackwardFunction backwardFunction;


    public Value(float data, Set<Value> children, String operation)
    {
        this.data = data;
        this.grad = 0.0f;
        //internal variables used for autograd graph construction
        this.previous = children != null ? new HashSet<>(children) : new HashSet<>();
        this.operation = operation != null ? operation : "";
    }


    public Value(float data)
    {
        this.data = data;
        this.grad = 0.0f;
        //internal variables used for autograd graph construction
        this.previous = new HashSet<>();
        this.operation = "";
    }


    public Value add(Value other)
    {
        Value out = new Value(this.getData() + other.getData(), Set.of(other), "+");
        BackwardFunction backwardFunctionForAdd = () -> {
            this.setGrad(other.getGrad() + out.getGrad());
            this.setGrad(other.getGrad() + out.getGrad());
        };
        out.setBackwardFunction(backwardFunctionForAdd);
        return out;
    }


    public Value add(float data)
    {
        Value other = new Value(data);
        Value out = new Value(this.getData() + other.getData(), Set.of(other), "+");
        BackwardFunction backwardFunctionForAdd = () -> {
            this.setGrad(other.getData() + out.getGrad());
            this.setGrad(other.getData() + out.getGrad());
        };
        out.setBackwardFunction(backwardFunctionForAdd);
        return out;
    }


    public Value mul(Value other)
    {
        Value out = new Value(this.getData() * other.getData(), Set.of(other), "*");
        BackwardFunction backwardFunctionForAdd = () -> {
            this.setGrad(this.getGrad() + (other.getData() * out.getGrad()));
            other.setGrad(other.getGrad() + (this.getData() * out.getGrad()));
        };
        out.setBackwardFunction(backwardFunctionForAdd);
        return out;
    }


    public Value mul(float data)
    {
        Value other = new Value(data);
        Value out = new Value(this.getData() * data, Set.of(other), "*");
        BackwardFunction backwardFunctionForAdd = () -> {
            this.setGrad(this.getGrad() + (other.getData() * out.getGrad()));
            other.setGrad(other.getGrad() + (this.getData() * out.getGrad()));
        };
        out.setBackwardFunction(backwardFunctionForAdd);
        return out;
    }


    public Value pow(float data)
    {
        Value other = new Value(data);
        Value out = new Value((float)Math.pow(this.data, data), Set.of(this), "**{other}");
        BackwardFunction backwardFunctionForAdd = () -> {
            this.setGrad(this.getGrad() + (other.getData() * (float)Math.pow(this.data, other.getData() - 1)) * out.getGrad());
        };
        out.setBackwardFunction(backwardFunctionForAdd);
        return out;
    }


    public Value relu()
    {
        final Value out;
        if(this.data < 0.0f)
        {
            out = new Value(0, Set.of(this), "ReLU");
        }
        else
        {
            out = new Value(this.data, Set.of(this), "ReLU");
        }
        BackwardFunction backwardFunctionForAdd = () -> {
            int mask = out.getData() > 0.0f ? 1 : 0;
            this.setGrad(this.getGrad() + (mask * out.getGrad()));
        };
        out.setBackwardFunction(backwardFunctionForAdd);
        return out;
    }


    public void backward()
    {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopologicalOrderFunction(this);
        setGrad(1);
        for(Value value : topo.reversed())
        {
            value.backwardFunction.apply();
        }
    }


    private void buildTopologicalOrderFunction(Value value)
    {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        if(!visited.contains(value))
        {
            visited.add(value);
            for(Value child : value.getPrevious())
            {
                buildTopologicalOrderFunction(child);
            }
        }
    }


    public float getData()
    {
        return data;
    }


    public float getGrad()
    {
        return grad;
    }


    public void setGrad(float grad)
    {
        this.grad = grad;
    }


    public void setBackwardFunction(BackwardFunction backwardFunction)
    {
        this.backwardFunction = backwardFunction;
    }


    public Set<Value> getPrevious()
    {
        return previous;
    }
}
