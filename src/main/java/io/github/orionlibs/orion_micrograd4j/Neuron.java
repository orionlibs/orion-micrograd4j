package io.github.orionlibs.orion_micrograd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron extends Module
{
    private boolean nonlin;
    private List<Value> w;
    private Value b;


    public Neuron(int nin, boolean nonlin)
    {
        this.nonlin = nonlin;
        List<Value> wTemp = new ArrayList<>();
        for(int i = 0; i < nin; i++)
        {
            wTemp.add(new Value(new Random().nextFloat(1f)));
        }
        this.w = wTemp;
        this.b = new Value(0);
    }


    public Value run(List<Value> x)
    {
        Value product = new Value(0);
        for(int i = 0; i < x.size(); i++)
        {
            product.add(w.get(i).mul(x.get(i)));
        }
        Value act = product.add(0);
        return nonlin ? act.relu() : act;
    }


    @Override
    public List<Value> parameters()
    {
        List<Value> temp = List.copyOf(w);
        temp.add(b);
        return temp;
    }


    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append(nonlin ? "ReLU" : "Linear");
        sb.append("Neuron(");
        sb.append(w.size());
        sb.append(")");
        return sb.toString();
    }
}
