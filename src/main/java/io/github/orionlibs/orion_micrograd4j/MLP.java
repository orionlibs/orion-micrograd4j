package io.github.orionlibs.orion_micrograd4j;

import java.util.ArrayList;
import java.util.List;

public class MLP extends Module
{
    private List<Integer> sz;
    private List<Layer> layers;


    public MLP(int nin, List<Integer> nouts)
    {
        this.sz = new ArrayList<>();
        sz.add(nin);
        sz.addAll(nouts);
        layers = new ArrayList<>();
        for(int i = 0; i < nouts.size(); i++)
        {
            layers.add(new Layer(sz.get(i), sz.get(i + 1), i != nouts.size() - 1));
        }
    }


    public List<Value> run(List<Value> x)
    {
        List<Value> values = List.copyOf(x);
        for(Layer layer : layers)
        {
            values = layer.run(values);
        }
        return values;
    }


    @Override
    public List<Value> parameters()
    {
        List<Value> temp = new ArrayList<>();
        for(Layer layer : layers)
        {
            temp.addAll(layer.parameters());
        }
        return temp;
    }


    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("MLP of ");
        for(int i = 0; i < layers.size(); i++)
        {
            sb.append(layers.get(i));
            if(i < layers.size() - 1)
            {
                sb.append(", ");
            }
        }
        return sb.toString();
    }
}
