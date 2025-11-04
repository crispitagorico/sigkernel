import torch
import sigkernel


def test_mps_availability():
    """Test that MPS is available on this system."""
    if not torch.backends.mps.is_available():
        print("SKIP: MPS not available on this system")
        return False
    print("✓ MPS is available")
    return True


def test_basic_kernel():
    """Test basic batch kernel computation: MPS vs CPU."""
    print("\nTesting basic kernel computation...")

    X_cpu = torch.randn(3, 8, 2, dtype=torch.float32)
    Y_cpu = torch.randn(3, 10, 2, dtype=torch.float32)

    X_mps = X_cpu.to('mps')
    Y_mps = Y_cpu.to('mps')

    static_kernel = sigkernel.RBFKernel(sigma=1.0)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    K_cpu = sig_kernel.compute_kernel(X_cpu, Y_cpu)
    K_mps = sig_kernel.compute_kernel(X_mps, Y_mps)

    K_mps_on_cpu = K_mps.cpu()

    if torch.allclose(K_mps_on_cpu, K_cpu, rtol=1e-4, atol=1e-5):
        print(f"✓ Basic kernel: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e}")
        return True
    else:
        print(f"✗ Basic kernel: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e} (FAILED)")
        return False


def test_gram_matrix():
    """Test Gram matrix computation: MPS vs CPU."""
    print("\nTesting Gram matrix computation...")

    X_cpu = torch.randn(4, 6, 2, dtype=torch.float32)
    Y_cpu = torch.randn(5, 6, 2, dtype=torch.float32)

    X_mps = X_cpu.to('mps')
    Y_mps = Y_cpu.to('mps')

    static_kernel = sigkernel.RBFKernel(sigma=0.5)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    G_cpu = sig_kernel.compute_Gram(X_cpu, Y_cpu, sym=False)
    G_mps = sig_kernel.compute_Gram(X_mps, Y_mps, sym=False)

    G_mps_on_cpu = G_mps.cpu()

    if torch.allclose(G_mps_on_cpu, G_cpu, rtol=1e-4, atol=1e-5):
        print(f"✓ Gram matrix: max diff = {torch.max(torch.abs(G_mps_on_cpu - G_cpu)).item():.2e}")
        return True
    else:
        print(f"✗ Gram matrix: max diff = {torch.max(torch.abs(G_mps_on_cpu - G_cpu)).item():.2e} (FAILED)")
        return False


def test_symmetric_gram():
    """Test symmetric Gram matrix computation: MPS vs CPU."""
    print("\nTesting symmetric Gram matrix...")

    X_cpu = torch.randn(5, 8, 2, dtype=torch.float32)

    X_mps = X_cpu.to('mps')

    static_kernel = sigkernel.RBFKernel(sigma=1.0)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    G_cpu = sig_kernel.compute_Gram(X_cpu, X_cpu, sym=True)
    G_mps = sig_kernel.compute_Gram(X_mps, X_mps, sym=True)

    G_mps_on_cpu = G_mps.cpu()

    if torch.allclose(G_mps_on_cpu, G_cpu, rtol=1e-4, atol=1e-5):
        print(f"✓ Symmetric Gram: max diff = {torch.max(torch.abs(G_mps_on_cpu - G_cpu)).item():.2e}")
        return True
    else:
        print(f"✗ Symmetric Gram: max diff = {torch.max(torch.abs(G_mps_on_cpu - G_cpu)).item():.2e} (FAILED)")
        return False


def test_gradients():
    """Test gradient computation via backward pass: MPS."""
    print("\nTesting gradient computation...")

    X_mps = torch.randn(3, 6, 2, device='mps', dtype=torch.float32, requires_grad=True)
    Y_mps = torch.randn(3, 6, 2, device='mps', dtype=torch.float32)

    static_kernel = sigkernel.RBFKernel(sigma=0.5)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    K = sig_kernel.compute_kernel(X_mps, Y_mps)
    loss = K.sum()
    loss.backward()

    if X_mps.grad is not None and torch.isfinite(X_mps.grad).all():
        print(f"✓ Gradients: grad norm = {torch.norm(X_mps.grad).item():.2e}")
        return True
    else:
        print("✗ Gradients: contains NaN or inf (FAILED)")
        return False


def test_dyadic_order():
    """Test with different dyadic orders: MPS vs CPU."""
    print("\nTesting dyadic_order=1...")

    X_cpu = torch.randn(2, 5, 2, dtype=torch.float32)
    Y_cpu = torch.randn(2, 5, 2, dtype=torch.float32)

    X_mps = X_cpu.to('mps')
    Y_mps = Y_cpu.to('mps')

    static_kernel = sigkernel.RBFKernel(sigma=1.0)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=1)

    K_cpu = sig_kernel.compute_kernel(X_cpu, Y_cpu)
    K_mps = sig_kernel.compute_kernel(X_mps, Y_mps)

    K_mps_on_cpu = K_mps.cpu()

    if torch.allclose(K_mps_on_cpu, K_cpu, rtol=1e-4, atol=1e-5):
        print(f"✓ Dyadic order 1: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e}")
        return True
    else:
        print(f"✗ Dyadic order 1: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e} (FAILED)")
        return False


def test_mmd():
    """Test MMD computation: MPS vs CPU."""
    print("\nTesting MMD computation...")

    X_cpu = torch.randn(6, 8, 2, dtype=torch.float32)
    Y_cpu = torch.randn(6, 8, 2, dtype=torch.float32)

    X_mps = X_cpu.to('mps')
    Y_mps = Y_cpu.to('mps')

    static_kernel = sigkernel.RBFKernel(sigma=0.5)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    mmd_cpu = sig_kernel.compute_mmd(X_cpu, Y_cpu)
    mmd_mps = sig_kernel.compute_mmd(X_mps, Y_mps)

    mmd_mps_val = mmd_mps.cpu().item()
    mmd_cpu_val = mmd_cpu.item()

    if abs(mmd_mps_val - mmd_cpu_val) < 1e-4:
        print(f"✓ MMD: diff = {abs(mmd_mps_val - mmd_cpu_val):.2e}")
        return True
    else:
        print(f"✗ MMD: diff = {abs(mmd_mps_val - mmd_cpu_val):.2e} (FAILED)")
        return False


def test_linear_kernel():
    """Test with LinearKernel: MPS vs CPU."""
    print("\nTesting with LinearKernel...")

    X_cpu = torch.randn(3, 6, 2, dtype=torch.float32)
    Y_cpu = torch.randn(3, 6, 2, dtype=torch.float32)

    X_mps = X_cpu.to('mps')
    Y_mps = Y_cpu.to('mps')

    static_kernel = sigkernel.LinearKernel()
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    K_cpu = sig_kernel.compute_kernel(X_cpu, Y_cpu)
    K_mps = sig_kernel.compute_kernel(X_mps, Y_mps)

    K_mps_on_cpu = K_mps.cpu()

    if torch.allclose(K_mps_on_cpu, K_cpu, rtol=1e-4, atol=1e-5):
        print(f"✓ LinearKernel: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e}")
        return True
    else:
        print(f"✗ LinearKernel: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e} (FAILED)")
        return False


def test_asymmetric_paths():
    """Test with different path lengths: MPS vs CPU."""
    print("\nTesting asymmetric path lengths...")

    X_cpu = torch.randn(2, 10, 2, dtype=torch.float32)
    Y_cpu = torch.randn(2, 15, 2, dtype=torch.float32)

    X_mps = X_cpu.to('mps')
    Y_mps = Y_cpu.to('mps')

    static_kernel = sigkernel.RBFKernel(sigma=1.0)
    sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

    K_cpu = sig_kernel.compute_kernel(X_cpu, Y_cpu)
    K_mps = sig_kernel.compute_kernel(X_mps, Y_mps)

    K_mps_on_cpu = K_mps.cpu()

    if torch.allclose(K_mps_on_cpu, K_cpu, rtol=1e-4, atol=1e-5):
        print(f"✓ Asymmetric paths: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e}")
        return True
    else:
        print(f"✗ Asymmetric paths: max diff = {torch.max(torch.abs(K_mps_on_cpu - K_cpu)).item():.2e} (FAILED)")
        return False


def run_all_tests():
    """Run all MPS tests."""
    print("=" * 60)
    print("MPS Backend Test Suite for sigkernel")
    print("=" * 60)

    if not test_mps_availability():
        print("\nTests skipped: MPS not available")
        return

    tests = [
        test_basic_kernel,
        test_gram_matrix,
        test_symmetric_gram,
        test_gradients,
        test_dyadic_order,
        test_mmd,
        test_linear_kernel,
        test_asymmetric_paths,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Exception - {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
