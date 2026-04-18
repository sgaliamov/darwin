# Genetic Algorithm

## Design Choices

### Uniqueness

When using floating-point numbers as genes, individuals will always be slightly different, even when these differences don't affect the evaluation results.

Here are our options:

1. **Delegate** gene definition to the client \
   Simply defining a gene type wouldn't provide much value to the client.
   It would only make sense if clients could define their own `Individual` type.
   However, this would require clients to implement all operations (mutation, random generation, crossover),
   making the GA framework more complex and error-prone to use.

2. Define **precision** for floating-point numbers \
   This is a simple solution requiring minimal refactoring.
   For example, with a maximum value of 500, we could use a precision of 0.001 (500 * 0.0009 = 0.45).
   We will have to hardcode it for now to not oven-complicate the API. And there is the better option.

3. Use **integer** type for genes \
   This is likely the best option as it improves calculation speed and provides a direct mapping between gene and actual values.
   This approach allows defining gene ranges directly in GA settings.
   Instead of using `genomeLength`, we can specify `genome: [500, 200, 200]`, where numbers represent maximum values.
   It will also simplify domain specific configurations, as all ranges will go to `genome` field.

### Elite

When some individual starts dominate in a pool, the whole pool can become polluted with its mutants.
To mitigate it we need create more pools, and during crossover direct all children in one direction.

### Ideas ot improve

1. **Graveyard** - to cache dead individuals to not process them again. Reset it when GA reused with new set of data.
To avoid re-evaluating identical individuals, we need to maintain uniqueness in the population and potentially store some individuals in a "graveyard" collection.
