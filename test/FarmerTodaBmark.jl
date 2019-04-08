
# test against Farmer Toda MATLAB values


hex2num_me(s) = reinterpret(Float64, parse(UInt64, s, base = 16))

DFT99 = dropdims(hex2num_me.(["c0340cd923b15b89"   "c02e1345b58a094e"   "c0240cd923b15b89"   "c0140cd923b15b88"   "0000000000000000"   "40140cd923b15b88"   "40240cd923b15b8a"   "402e1345b58a094e"   "40340cd923b15b89"]); dims=1)
PFT99 = hex2num_me.(
      [ "3feebdff4a767113"   "3fa3c61f93aef8f1"   "3f464066ed540f5d"   "3edd4507c34018f2"   "3dded173035783dd"   "3c49f92e9f402b9c"   "3a21859dbed5564e"   "3762ec61a6257787"   "34105bcf2cf90500";
        "3d99649e3e85a1e9"   "3fef2e561795e507"   "3f96a88e64e8b5cd"   "3f64ba2e346747b5"   "3f4e8d7202f75e32"   "3ecfa9d74ae8221e"   "3de7124bc92f5c91"   "3c97a393256534a1"   "3ae10768d4a1c15f";
        "3c1af68f17efc8f0"   "3dc8b97cb927921b"   "3fef97050c545fe9"   "3f814097bd0adae2"   "3f5c24ab3f85ffdf"   "3f64e95cfe5cc002"   "3f1f902982d4999c"   "3e883178a38482b3"   "3da2d5af6cc1ee2f";
        "3c65cee5b29700db"   "3d2b083ffb64d194"   "3e3ef0b49f17ce6a"   "3fefe89723f18a15"   "3f3e2eb32f62d91c"   "3f12d4fab284def1"   "3f41eede5c549b17"   "3f529bdeda0b2764"   "3f450acbc309a532";
        "3f5444b0bd52b3ea"   "3ee6ebb33b5859ca"   "3e9906756359ae1f"   "3edcc643e8f3b048"   "3fefeb6f21308b9c"   "3edcc643e93f9d2c"   "3e99067563ddbec7"   "3ee6ebb33c0dc9f9"   "3f5444b0be28a035";
        "3f450acbd076ef71"   "3f529bdec9630975"   "3f41eede3d641bd9"   "3f12d4fa8fa25180"   "3f3e2eb308db0eb2"   "3fefe897240426f9"   "3e3ef0b4dc7bb8a6"   "3d2b0840799bcb9d"   "3c65cee6624c25a4";
        "3da2d5af6d4add66"   "3e883178a3f70f30"   "3f1f9029832ab837"   "3f64e95cfe76ec17"   "3f5c24ab3f8e8e0a"   "3f814097bd08e239"   "3fef97050c5446a5"   "3dc8b97cb9375469"   "3c1af68f182082f7";
        "3ae10768d5312c4c"   "3c97a39325f2c4a0"   "3de7124bc98ad89f"   "3ecfa9d74b3299b9"   "3f4e8d72031a74db"   "3f64ba2e346ea424"   "3f96a88e64e812c8"   "3fef2e561795d9ea"   "3d99649e3e90b030";
        "34105bd6cc1238fd"   "3762ec6842abd059"   "3a2185a21e0a922f"   "3c49f932f0f81e51"   "3dded1761626dc19"   "3edd450938876835"   "3f4640674ba35ba6"   "3fa3c61f936a7347"   "3feebdff4a626af6"]
      )

# setup
ρ = 0.99
n = 9
L = 2
σ = 1.

# set up state space
σlr = sqrt(1.0 / (1.0-ρ^2))
nσ = sqrt(n-1)
y = range(-σlr*nσ, stop=σlr*nσ, length=n)
@test y ≈ DFT99

# output
P, JN, Λ, numMoments, approxErr = MarkovTransitionMatrices.discreteNormalApprox(y, y, (x::Real, st::Real) -> x - ρ*st, 2)
@test maximum(abs.(P .- PFT99)) < 1e-7
@test maximum(abs.(0.5 * (P .- PFT99) ./ (P .+ PFT99))) < 1e-5













#
