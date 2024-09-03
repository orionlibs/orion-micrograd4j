package io.github.orionlibs.orion_micrograd4j;

import java.util.ArrayList;
import java.util.List;

public class Module
{
    public List<Value> parameters()
    {
        return new ArrayList<>();
    }


    public void zeroGrad()
    {
        for(Value param : parameters())
        {
            param.setGrad(0);
        }
    }
}
