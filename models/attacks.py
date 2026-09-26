import torchattacks

def get_attack(model, attack_type="pgd", epsilon=0.03):
    if attack_type == "fgsm":
        return torchattacks.FGSM(model, eps=epsilon)
    elif attack_type == "pgd":
        return torchattacks.PGD(model, eps=epsilon, alpha=epsilon / 4, steps=10)
    elif attack_type == "cw":
        return torchattacks.CW(model, c=1, steps=100)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")