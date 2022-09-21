# Individual Code Snippets

## Idea
The idea is that the input program code should be divided into individual, not further separable elements, in a sense that the code cannot be further separated without breaking highly coupled code (e.g. a for-loop should be kept together). The program parts can then be translated individually and checked for their correctness, in which the output for given inputs is examined. From the individual translated parts one should be able to assemble the entire program.

## Experiments
Run the following command to see an example translation where each line was translated individually.

```sh
python -m codegen_sources.scripts.snippets.translate
```