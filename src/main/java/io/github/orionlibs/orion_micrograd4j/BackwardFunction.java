package io.github.orionlibs.orion_micrograd4j;

@FunctionalInterface
public interface BackwardFunction
{
    void apply();
}
