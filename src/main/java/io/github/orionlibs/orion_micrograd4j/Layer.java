package io.github.orionlibs.orion_micrograd4j;

import java.util.ArrayList;
import java.util.List;

public class Layer extends Module
{
    private List<Neuron> neurons;


    public Layer(int nin, int nout, boolean nonlin)
    {
        this.neurons = new ArrayList<>();
        for(int i = 0; i < nout; i++)
        {
            neurons.add(new Neuron(nin, nonlin));
        }
    }


    public List<Value> run(List<Value> x)
    {
        List<Value> values = new ArrayList<>();
        for(Neuron neuron : neurons)
        {
            values.add(neuron.run(x));
        }
        return values;
    }


    @Override
    public List<Value> parameters()
    {
        List<Value> temp = new ArrayList<>();
        for(Neuron neuron : neurons)
        {
            temp.addAll(neuron.parameters());
        }
        return temp;
    }


    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("Layer of ");
        for(int i = 0; i < neurons.size(); i++)
        {
            sb.append(neurons.get(i));
            if(i < neurons.size() - 1)
            {
                sb.append(", ");
            }
        }
        return sb.toString();
    }
}
