#ifndef GRAPHBREW_REORDER_EXPERIMENTAL_SPECTRAL_BLOCKS_H_
#define GRAPHBREW_REORDER_EXPERIMENTAL_SPECTRAL_BLOCKS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common.h"

namespace graphbrew::experimental {

inline constexpr size_t kSpectralMaxTotalVertices = 256;
inline constexpr size_t kSpectralMaxComponentVertices = 64;
inline constexpr size_t kSpectralMaxSweeps = 128;
inline constexpr double kSpectralTolerance = 1e-13;

enum class SpectralOrderStatus {
    Success,
    Trivial,
    TooLarge,
    Degenerate,
    NonConverged,
};

template <typename K>
struct SpectralOrderResult {
    SpectralOrderStatus status = SpectralOrderStatus::Trivial;
    std::vector<K> order;
    size_t componentCount = 0;
    // Eigenvalues describe the first nontrivial component in base order.
    double lambda2 = 0.0;
    double lambda3 = 0.0;
    double minimumEigengap = 0.0;
    // Residual is the maximum backward residual over solved components.
    double residual = 0.0;
};

namespace spectral_detail {

struct EigenResult {
    bool converged = false;
    std::vector<double> values;
    std::vector<double> vectors;
};

inline double offDiagonalNorm(
    const std::vector<double>& matrix,
    size_t size)
{
    double squared = 0.0;
    for (size_t row = 0; row < size; ++row) {
        for (size_t column = 0; column < size; ++column) {
            if (row == column) continue;
            squared +=
                matrix[row * size + column]
                * matrix[row * size + column];
        }
    }
    return std::sqrt(squared);
}

inline EigenResult jacobiEigen(
    std::vector<double> matrix,
    size_t size,
    size_t maxSweeps,
    double tolerance)
{
    EigenResult result;
    result.vectors.assign(size * size, 0.0);
    for (size_t i = 0; i < size; ++i)
        result.vectors[i * size + i] = 1.0;

    for (size_t sweep = 0; sweep < maxSweeps; ++sweep) {
        for (size_t p = 0; p < size; ++p) {
            for (size_t q = p + 1; q < size; ++q) {
                const double apq = matrix[p * size + q];
                const double app = matrix[p * size + p];
                const double aqq = matrix[q * size + q];
                const double rotationScale =
                    std::sqrt(std::abs(app))
                    * std::sqrt(std::abs(aqq));
                if (
                    std::abs(apq)
                    <= tolerance * rotationScale
                ) {
                    continue;
                }
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = std::copysign(
                    1.0
                        / (
                            std::abs(tau)
                            + std::sqrt(1.0 + tau * tau)
                        ),
                    tau);
                const double cosine = 1.0 / std::sqrt(1.0 + t * t);
                const double sine = t * cosine;

                for (size_t k = 0; k < size; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = matrix[k * size + p];
                    const double akq = matrix[k * size + q];
                    const double nextKp =
                        cosine * akp - sine * akq;
                    const double nextKq =
                        sine * akp + cosine * akq;
                    matrix[k * size + p] =
                        matrix[p * size + k] = nextKp;
                    matrix[k * size + q] =
                        matrix[q * size + k] = nextKq;
                }

                matrix[p * size + p] =
                    app - t * apq;
                matrix[q * size + q] =
                    aqq + t * apq;
                matrix[p * size + q] =
                    matrix[q * size + p] = 0.0;

                for (size_t k = 0; k < size; ++k) {
                    const double vkp =
                        result.vectors[k * size + p];
                    const double vkq =
                        result.vectors[k * size + q];
                    result.vectors[k * size + p] =
                        cosine * vkp - sine * vkq;
                    result.vectors[k * size + q] =
                        sine * vkp + cosine * vkq;
                }
            }
        }

        double diagonalMagnitude = 0.0;
        for (size_t i = 0; i < size; ++i) {
            diagonalMagnitude += std::abs(matrix[i * size + i]);
        }
        if (
            offDiagonalNorm(matrix, size)
            <= tolerance
                * std::max(1.0, diagonalMagnitude)
        ) {
            result.converged = true;
            break;
        }
    }

    result.values.resize(size);
    for (size_t i = 0; i < size; ++i)
        result.values[i] = matrix[i * size + i];
    return result;
}

}  // namespace spectral_detail

template <typename K>
SpectralOrderResult<K> exactFiedlerBlockOrder(
    const std::vector<std::vector<std::pair<K, double>>>& adjacency,
    const std::vector<K>& baseOrder = {},
    const std::vector<size_t>& vertexMass = {})
{
    const size_t vertexCount = adjacency.size();
    SpectralOrderResult<K> result;
    if (vertexCount > kSpectralMaxTotalVertices) {
        result.status = SpectralOrderStatus::TooLarge;
        return result;
    }
    if (
        !baseOrder.empty()
        && baseOrder.size() != vertexCount
    ) {
        throw std::invalid_argument(
            "Spectral base order must cover every vertex");
    }

    std::vector<K> effectiveBase = baseOrder;
    if (effectiveBase.empty()) {
        effectiveBase.resize(vertexCount);
        std::iota(effectiveBase.begin(), effectiveBase.end(), K(0));
    }
    const std::vector<K> baseRank = invertOrder(effectiveBase);

    std::vector<size_t> mass = vertexMass;
    if (mass.empty()) mass.assign(vertexCount, 1);
    if (mass.size() != vertexCount) {
        throw std::invalid_argument(
            "Spectral vertex masses must cover every vertex");
    }
    if (vertexCount == 0) {
        result.status = SpectralOrderStatus::Trivial;
        return result;
    }

    const size_t matrixEntries = checkedSizeMultiply(
        vertexCount,
        vertexCount,
        "Spectral matrix size overflowed");
    std::vector<double> weights(matrixEntries, 0.0);
    for (size_t source = 0; source < vertexCount; ++source) {
        for (const auto& [targetValue, weight] : adjacency[source]) {
            const size_t target = checkedIndex(
                targetValue,
                vertexCount,
                "Spectral adjacency target is out of range");
            if (
                !std::isfinite(weight)
                || weight < 0.0
            ) {
                throw std::invalid_argument(
                    "Spectral adjacency contains invalid data");
            }
            if (target == source || weight == 0.0) continue;
            double& aggregate =
                weights[source * vertexCount + target];
            aggregate += weight;
            if (!std::isfinite(aggregate)) {
                throw std::overflow_error(
                    "Spectral edge-weight sum overflowed");
            }
        }
    }
    for (size_t left = 0; left < vertexCount; ++left) {
        for (size_t right = left + 1; right < vertexCount; ++right) {
            const double forward =
                weights[left * vertexCount + right];
            const double reverse =
                weights[right * vertexCount + left];
            if ((forward == 0.0) != (reverse == 0.0)) {
                throw std::invalid_argument(
                    "Spectral adjacency support must be symmetric");
            }
            const double scale =
                std::max(std::abs(forward), std::abs(reverse));
            if (
                std::abs(forward - reverse)
                > kSpectralTolerance * scale
            ) {
                throw std::invalid_argument(
                    "Spectral adjacency must be symmetric");
            }
            const double canonical =
                forward + (reverse - forward) / 2.0;
            if (!std::isfinite(canonical)) {
                throw std::overflow_error(
                    "Spectral canonical edge weight overflowed");
            }
            weights[left * vertexCount + right] = canonical;
            weights[right * vertexCount + left] = canonical;
        }
        if (mass[left] == 0) {
            for (size_t right = 0; right < vertexCount; ++right) {
                if (weights[left * vertexCount + right] != 0.0) {
                    throw std::invalid_argument(
                        "Empty spectral vertices cannot have edges");
                }
            }
        }
    }

    std::vector<char> visited(vertexCount, 0);
    std::vector<std::vector<K>> components;
    std::vector<K> emptyVertices;
    for (K vertexValue : effectiveBase) {
        const size_t vertex = static_cast<size_t>(vertexValue);
        if (mass[vertex] == 0) {
            visited[vertex] = 1;
            emptyVertices.push_back(vertexValue);
            continue;
        }
        if (visited[vertex]) continue;
        std::queue<size_t> frontier;
        frontier.push(vertex);
        visited[vertex] = 1;
        components.emplace_back();
        while (!frontier.empty()) {
            const size_t current = frontier.front();
            frontier.pop();
            components.back().push_back(static_cast<K>(current));
            for (size_t neighbor = 0;
                 neighbor < vertexCount;
                 ++neighbor) {
                if (
                    weights[current * vertexCount + neighbor] > 0.0
                    && mass[neighbor] > 0
                    && !visited[neighbor]
                ) {
                    visited[neighbor] = 1;
                    frontier.push(neighbor);
                }
            }
        }
    }

    std::stable_sort(
        components.begin(), components.end(),
        [&](const auto& left, const auto& right) {
            size_t leftRank = vertexCount;
            size_t rightRank = vertexCount;
            for (K vertex : left) {
                leftRank = std::min(
                    leftRank,
                    static_cast<size_t>(baseRank[vertex]));
            }
            for (K vertex : right) {
                rightRank = std::min(
                    rightRank,
                    static_cast<size_t>(baseRank[vertex]));
            }
            return leftRank < rightRank;
        });
    for (auto& component : components) {
        std::stable_sort(
            component.begin(), component.end(),
            [&](K left, K right) {
                return baseRank[left] < baseRank[right];
            });
    }

    result.componentCount = components.size();
    result.order.reserve(vertexCount);
    bool solvedSpectralComponent = false;
    bool recordedEigenvalues = false;
    double firstLambda2 = 0.0;
    double firstLambda3 = 0.0;
    double minimumEigengap =
        std::numeric_limits<double>::infinity();
    double maximumResidual = 0.0;
    auto fail = [&](SpectralOrderStatus status) {
        result.status = status;
        result.order.clear();
        result.lambda2 = 0.0;
        result.lambda3 = 0.0;
        result.minimumEigengap = 0.0;
        result.residual = 0.0;
        return result;
    };
    for (const auto& component : components) {
        const size_t size = component.size();
        if (size > kSpectralMaxComponentVertices)
            return fail(SpectralOrderStatus::TooLarge);
        if (size <= 2) {
            result.order.insert(
                result.order.end(),
                component.begin(),
                component.end());
            continue;
        }
        solvedSpectralComponent = true;

        const size_t componentEntries = checkedSizeMultiply(
            size,
            size,
            "Spectral component matrix size overflowed");
        std::vector<double> laplacian(componentEntries, 0.0);
        double maximumDegree = 0.0;
        for (size_t local = 0; local < size; ++local) {
            const size_t source =
                static_cast<size_t>(component[local]);
            double degree = 0.0;
            for (size_t targetLocal = 0;
                 targetLocal < size;
                 ++targetLocal) {
                const size_t target =
                    static_cast<size_t>(component[targetLocal]);
                const double weight =
                    weights[source * vertexCount + target];
                degree += weight;
                if (!std::isfinite(degree)) {
                    throw std::overflow_error(
                        "Spectral component degree overflowed");
                }
                laplacian[local * size + targetLocal] = -weight;
            }
            laplacian[local * size + local] = degree;
            maximumDegree = std::max(maximumDegree, degree);
        }
        if (maximumDegree <= 0.0 || !std::isfinite(maximumDegree))
            return fail(SpectralOrderStatus::Degenerate);
        for (double& value : laplacian)
            value /= maximumDegree;

        const auto eigen = spectral_detail::jacobiEigen(
            laplacian,
            size,
            kSpectralMaxSweeps,
            kSpectralTolerance);
        if (!eigen.converged)
            return fail(SpectralOrderStatus::NonConverged);

        std::vector<size_t> eigenOrder(size);
        std::iota(eigenOrder.begin(), eigenOrder.end(), 0);
        std::stable_sort(
            eigenOrder.begin(), eigenOrder.end(),
            [&](size_t left, size_t right) {
                return eigen.values[left] < eigen.values[right];
            });
        const size_t fiedlerColumn = eigenOrder[1];
        const double lambda1 = eigen.values[eigenOrder[0]];
        const double lambda2 = eigen.values[eigenOrder[1]];
        const double lambda3 = eigen.values[eigenOrder[2]];

        std::vector<double> fiedler(size);
        double norm = 0.0;
        double maximum = 0.0;
        double sum = 0.0;
        for (size_t local = 0; local < size; ++local) {
            fiedler[local] =
                eigen.vectors[local * size + fiedlerColumn];
            norm += fiedler[local] * fiedler[local];
            maximum = std::max(maximum, std::abs(fiedler[local]));
            sum += fiedler[local];
        }
        norm = std::sqrt(norm);
        if (norm == 0.0 || maximum == 0.0)
            return fail(SpectralOrderStatus::Degenerate);

        double residualSquared = 0.0;
        double laplacianNormSquared = 0.0;
        for (size_t row = 0; row < size; ++row) {
            double action = 0.0;
            for (size_t column = 0; column < size; ++column) {
                laplacianNormSquared +=
                    laplacian[row * size + column]
                    * laplacian[row * size + column];
                action +=
                    laplacian[row * size + column]
                    * fiedler[column];
            }
            const double residual =
                action - lambda2 * fiedler[row];
            residualSquared += residual * residual;
        }
        const double denominator =
            (std::sqrt(laplacianNormSquared) + std::abs(lambda2))
            * norm;
        const double residual =
            std::sqrt(residualSquared)
            / std::max(denominator, std::numeric_limits<double>::min());
        const double numericalError = std::max(
            1000.0 * kSpectralTolerance,
            10.0 * residual);
        if (
            residual > 1e-10
            || std::abs(lambda1) > numericalError
            || lambda2 - lambda1 <= numericalError
            || lambda3 - lambda2 <= numericalError
            || std::abs(sum) > 1e-8 * norm * std::sqrt(size)
        ) {
            return fail(
                residual > 1e-10
                    ? SpectralOrderStatus::NonConverged
                    : SpectralOrderStatus::Degenerate);
        }

        double orientation = 0.0;
        double meanRank = 0.0;
        for (K vertex : component)
            meanRank += baseRank[vertex];
        meanRank /= static_cast<double>(size);
        for (size_t local = 0; local < size; ++local) {
            orientation += fiedler[local] * (
                static_cast<double>(baseRank[component[local]])
                - meanRank);
        }
        double rankNormSquared = 0.0;
        for (K vertex : component) {
            const double centered =
                static_cast<double>(baseRank[vertex]) - meanRank;
            rankNormSquared += centered * centered;
        }
        const double orientationTolerance =
            1e-10 * norm * std::sqrt(rankNormSquared);
        if (std::abs(orientation) <= orientationTolerance) {
            size_t anchor = 0;
            for (size_t local = 1; local < size; ++local) {
                if (
                    std::abs(fiedler[local])
                    > std::abs(fiedler[anchor]) + numericalError
                ) {
                    anchor = local;
                }
            }
            if (fiedler[anchor] < 0.0) {
                for (double& value : fiedler) value = -value;
            }
        } else if (orientation < 0.0) {
            for (double& value : fiedler) value = -value;
        }

        const double fiedlerGap = std::min(
            lambda2 - lambda1,
            lambda3 - lambda2);
        minimumEigengap = std::min(
            minimumEigengap, fiedlerGap);
        maximumResidual = std::max(
            maximumResidual, residual);
        const double quantum = std::max(
            1e-12 * maximum,
            residual / fiedlerGap * maximum * 10.0);
        std::vector<size_t> localOrder(size);
        std::iota(localOrder.begin(), localOrder.end(), 0);
        std::stable_sort(
            localOrder.begin(), localOrder.end(),
            [&](size_t left, size_t right) {
                const int64_t leftKey = std::llround(
                    fiedler[left] / quantum);
                const int64_t rightKey = std::llround(
                    fiedler[right] / quantum);
                if (leftKey != rightKey) return leftKey < rightKey;
                const K leftVertex = component[left];
                const K rightVertex = component[right];
                if (baseRank[leftVertex] != baseRank[rightVertex]) {
                    return baseRank[leftVertex]
                        < baseRank[rightVertex];
                }
                return leftVertex < rightVertex;
            });
        for (size_t local : localOrder)
            result.order.push_back(component[local]);

        if (!recordedEigenvalues) {
            firstLambda2 = lambda2;
            firstLambda3 = lambda3;
            recordedEigenvalues = true;
        }
    }

    result.order.insert(
        result.order.end(),
        emptyVertices.begin(),
        emptyVertices.end());
    result.lambda2 = firstLambda2;
    result.lambda3 = firstLambda3;
    result.minimumEigengap = solvedSpectralComponent
        ? minimumEigengap : 0.0;
    result.residual = maximumResidual;
    result.status = solvedSpectralComponent
        ? SpectralOrderStatus::Success
        : SpectralOrderStatus::Trivial;
    return result;
}

}  // namespace graphbrew::experimental

#endif  // GRAPHBREW_REORDER_EXPERIMENTAL_SPECTRAL_BLOCKS_H_
