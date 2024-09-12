"""
Script to run the main function with different parameters

The main training function in main.py can be run with different parameters to compare the results
This can be achieved by using keyword arguments in the main function call

This code can be changed to sweep over different sets of parameters.
Any default parameter in main.py can be overridden by passing the parameter as a keyword argument in the main function call
The format is main(param1=value1, param2=value2, ...), where param1, param2, ... are the parameter names in main.py in small case
"""

import numpy as np
from rich.console import Console
from rich.table import Table

from main import main

GPU_ID = 0

if __name__ == '__main__':
    console = Console()
    table = Table(title='Results')
    table.add_column('N Layers')
    table.add_column('Hidden Channels')
    table.add_column('Test Accuracy (Mean)')
    table.add_column('Test Accuracy (Std)')
    table.add_column('Test Loss (Min)')
    table.add_column('Test Loss (Mean)')
    table.add_column('Test Loss (Std)')
    table.add_column('Test Loss (Max)')
    table.add_column('Worst Loss (Min)')
    table.add_column('Worst Loss (Mean)')
    table.add_column('Worst Loss (Std)')
    table.add_column('Worst Loss (Max)')


    failed_cases = []

    for n_layers in range(1, 6):
        for hidden_channels in [16, 32, 64, 128, 256, 512]:
            accs = np.zeros(10)
            loss = np.zeros(10)
            worst_loss = np.zeros(10)
            for seed in range(10):
                try:
                    accs[seed], loss[seed], worst_loss[seed] = main(
                        seed=seed,
                        gpu_id=GPU_ID,
                        n_layers=n_layers,
                        hidden_channels=hidden_channels,
                        train=True,
                        test=False,
                        verbose_test=False)
                except Exception as e:
                    failed_cases.append((n_layers, hidden_channels, seed, e))
            table.add_row(
                str(n_layers),
                str(hidden_channels),
                f'{accs.mean()*100:.2f}',
                f'{accs.std()*100:.2f}',
                f'{loss.min():.4f}',
                f'{loss.mean():.4f}',
                f'{loss.std():.4f}',
                f'{loss.max():.4f}',
                f'{worst_loss.min():.4f}',
                f'{worst_loss.mean():.4f}',
                f'{worst_loss.std():.4f}',
                f'{worst_loss.max():.4f}'
            )

    console.print(table)
    for failed_case in failed_cases:
        print(failed_case)
