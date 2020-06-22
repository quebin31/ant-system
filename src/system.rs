use crate::utils::{ToCharIndex, ToDisplayPath};
use anyhow::Error;
use ndarray::{Array2, Ix2, ShapeBuilder};
use rand::{thread_rng, Rng};
use std::io::Write;

fn init_pheromone_matrix<S>(shape: S, value: f64) -> Array2<f64>
where
    S: ShapeBuilder<Dim = Ix2>,
{
    Array2::from_shape_fn(shape, |(i, j)| if i == j { 0.0 } else { value })
}

fn compute_visiblity_matrix(distances: &Array2<f64>) -> Array2<f64> {
    distances.mapv(|v| 1.0 / v)
}

fn compute_cost(solution: &[usize], distances: &Array2<f64>) -> f64 {
    solution
        .windows(2)
        .fold(0.0, |acc, edge| acc + distances[[edge[0], edge[1]]])
}

#[derive(Debug, Clone, Default)]
pub struct AntSystem {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub q: f64,

    pub size: usize,
    pub initial: usize,

    pub distances: Array2<f64>,
    pub visibility: Array2<f64>,
    pub pheromones: Array2<f64>,
}

pub struct AntProps {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub q: f64,
    pub initial_pheromone: f64,
    pub distances: Array2<f64>,
}

impl AntSystem {
    pub fn new(size: usize, initial: usize, props: AntProps) -> Self {
        let shape = props.distances.raw_dim();

        let pheromones = init_pheromone_matrix(shape, props.initial_pheromone);
        let visibility = compute_visiblity_matrix(&props.distances);
        let distances = props.distances;

        Self {
            alpha: props.alpha,
            beta: props.beta,
            rho: props.rho,
            q: props.q,
            size,
            initial,
            distances,
            visibility,
            pheromones,
        }
    }

    pub fn run<W: Write>(&mut self, out: &mut W) -> Result<Vec<(Vec<usize>, f64)>, Error> {
        let mut solutions = Vec::new();

        for ant in 0..self.size {
            let solution = self.build_solution(ant, out)?;
            solutions.push(solution);
        }

        let mut solutions_to_return = Vec::new();
        for (ant, solution) in solutions.iter().enumerate() {
            let cost = compute_cost(&solution, &self.distances);
            writeln!(
                out,
                "Hormiga {}: {} (costo: {})",
                ant + 1,
                solution.to_display_path()?,
                cost
            )?;
            solutions_to_return.push((solution.clone(), cost));
        }

        self.update_pheromones(&solutions, out)?;

        Ok(solutions_to_return)
    }
}

impl AntSystem {
    fn build_solution<W: Write>(&self, i: usize, out: &mut W) -> Result<Vec<usize>, Error> {
        let no_cities = self.visibility.shape()[0];

        let mut visited = Vec::new();
        visited.push(self.initial);

        writeln!(out, "Hormiga {}", i + 1)?;
        writeln!(out, "Ciudad inicial: {}", self.initial.to_char_index())?;
        while visited.len() != no_cities {
            let mut probs = Vec::new();
            let curr = *visited.last().expect("No cities visited?");

            let sum =
                (0..no_cities)
                    .filter(|city| !visited.contains(city))
                    .fold(0.0, |acc, city| {
                        let pheromone = self.pheromones[[curr, city]];
                        let visibility = self.visibility[[curr, city]];

                        acc + pheromone.powf(self.alpha) * visibility.powf(self.beta)
                    });

            for city in 0..no_cities {
                if visited.contains(&city) {
                    continue;
                }

                let pheromone = self.pheromones[[curr, city]].powf(self.alpha);
                let visibility = self.visibility[[curr, city]].powf(self.beta);
                let prod = pheromone * visibility;
                let prob = prod / sum;

                writeln!(
                    out,
                    "{} -> {}: 𝜏^𝛼 = {}, 𝜂^𝛽 = {}, (𝜏^𝛼) * (𝜂^𝛽) = {}",
                    curr.to_char_index(),
                    city.to_char_index(),
                    pheromone,
                    visibility,
                    prod
                )?;

                probs.push((city, prob));
            }

            writeln!(out, "Suma: {}", sum)?;

            for (city, prob) in &probs {
                writeln!(
                    out,
                    "{} -> {}: prob = {}",
                    curr.to_char_index(),
                    city.to_char_index(),
                    prob
                )?;
            }

            let rand = thread_rng().gen_range(0., 1.);
            writeln!(out, "Número aleatorio: {}", rand)?;

            let (mut choosen, mut acc) = probs[0];
            for i in 0..probs.len() {
                if rand < acc || i == probs.len() - 1 {
                    choosen = probs[i].0;
                    break;
                }

                acc += probs[i + 1].1;
            }

            writeln!(out, "Siguiente ciudad: {}\n", choosen.to_char_index())?;
            visited.push(choosen);
        }

        writeln!(
            out,
            "Camino de la hormiga {}: {}\n---\n",
            i + 1,
            visited.to_display_path()?
        )?;

        Ok(visited)
    }

    fn update_pheromones<W: Write>(
        &mut self,
        solutions: &[Vec<usize>],
        out: &mut W,
    ) -> Result<(), Error> {
        let shape = self.pheromones.shape().to_owned();
        let costs: Vec<_> = solutions
            .iter()
            .map(|p| compute_cost(p, &self.distances))
            .collect();

        let edges: Vec<Vec<_>> = solutions
            .iter()
            .map(|p| p.windows(2).map(|edge| (edge[0], edge[1])).collect())
            .collect();

        for r in 0..shape[0] {
            for c in 0..shape[1] {
                let evaporation = self.rho * self.pheromones[[r, c]];
                write!(
                    out,
                    "{} -> {}: feromona = {} ",
                    r.to_char_index(),
                    c.to_char_index(),
                    evaporation
                )?;

                self.pheromones[[r, c]] = evaporation;

                for ant in 0..self.size {
                    if edges[ant].contains(&(r, c)) {
                        let w = self.q / costs[ant];
                        write!(out, "+ {} ", w)?;
                        self.pheromones[[r, c]] += w;
                    } else {
                        write!(out, "+ 0.0 ")?;
                    }
                }

                writeln!(out, "= {}", self.pheromones[[r, c]])?;
            }
        }

        Ok(())
    }
}
