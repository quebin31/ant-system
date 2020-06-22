pub mod system;
pub mod utils;

use crate::system::{AntProps, AntSystem};
use crate::utils::{pretty_matrix, ToDisplayPath};
use anyhow::Error;
use indicatif::ProgressIterator;
use ndarray::Array2;
use prettytable::format::consts::FORMAT_BOX_CHARS;
use prettytable::{cell, row, table};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Error> {
    let distances: Vec<_> = [
        0, 12, 3, 23, 1, 5, 23, 56, 12, 11, //
        12, 0, 9, 18, 3, 41, 45, 5, 41, 27, //
        3, 9, 0, 89, 56, 21, 12, 48, 14, 29, //
        23, 18, 89, 0, 87, 46, 75, 17, 50, 42, //
        1, 3, 56, 87, 0, 55, 22, 86, 14, 33, //
        5, 41, 21, 46, 55, 0, 21, 76, 54, 81, //
        23, 45, 12, 75, 22, 21, 0, 11, 57, 48, //
        56, 5, 48, 17, 86, 76, 11, 0, 63, 24, //
        12, 41, 14, 50, 14, 54, 57, 63, 0, 9, //
        11, 27, 29, 42, 33, 81, 48, 24, 9, 0, //
    ]
    .iter()
    .map(|v| *v as f64)
    .collect();

    let distances = Array2::from_shape_vec((10, 10), distances)?;

    let size = 10;
    let iters = 100;

    let props = AntProps {
        alpha: 1.0,
        beta: 1.0,
        rho: 0.99,
        q: 1.0,
        initial_pheromone: 0.1,
        distances,
    };

    let mut table = table! {
        ["Cantidad de hormigas", size],
        ["Cantidad de iteraciones", iters],
        ["Ciudad inicial", "A"],
        ["𝛼 (alpha)", props.alpha],
        ["𝛽 (beta)", props.beta],
        ["𝜌 (rho)", props.rho],
        ["Q", props.q],
        ["Feromona inicial", props.initial_pheromone]
    };
    table.set_format(*FORMAT_BOX_CHARS);

    let mut out = File::create("ant-system.out")?;

    writeln!(out, "Parámetros")?;
    writeln!(out, "{}\n", table)?;

    let mut ant_system = AntSystem::new(size, 3, props);
    let mut best = None;
    for i in (0..iters).progress() {
        writeln!(out, "------------------------------------")?;
        writeln!(out, "Iteración {}\n", i + 1)?;

        writeln!(
            out,
            "Matriz de visibilidad:\n{}",
            pretty_matrix(&ant_system.visibility, 6)
        )?;

        writeln!(
            out,
            "Matriz de feromonas:\n{}",
            pretty_matrix(&ant_system.pheromones, 6)
        )?;

        let solutions_w_costs = ant_system.run(&mut out)?;
        let min = solutions_w_costs
            .into_iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        writeln!(
            out,
            "Best path: {} with cost {}\n",
            min.0.to_display_path()?,
            min.1
        )?;

        best = Some(min);
    }

    let best = best.unwrap();
    writeln!(
        out,
        "\nBest global path: {} with cost {}",
        best.0.to_display_path()?,
        best.1
    )?;

    Ok(())
}
