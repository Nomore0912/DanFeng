#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


class Joint(object):
    def __init__(self, position, device='cpu', min_limit=None, max_limit=None):
        self.position = torch.tensor(position, device=device, dtype=torch.float32)
        self.min_limit = torch.tensor(min_limit, device=device, dtype=torch.float32) if min_limit is not None else None
        self.max_limit = torch.tensor(max_limit, device=device, dtype=torch.float32) if max_limit is not None else None

    def __repr__(self):
        return f"Joint(position={self.position.cpu().numpy()})"


class Skeleton(object):
    def __init__(self, joints):
        self.joints = joints
        self.device = joints[0].position.device
        self.distances = self.compute_distances()

    def compute_distances(self):
        return [torch.norm(self.joints[i].position - self.joints[i + 1].position) for i in range(len(self.joints) - 1)]

    def update_distances(self):
        self.distances = self.compute_distances()

    def __repr__(self):
        return f"Skeleton(joints={self.joints})"


class FABRIK(object):
    def __init__(self, skeleton, target, tolerance=1e-3, max_iterations=100):
        self.skeleton = skeleton
        self.target = torch.tensor(target, device=self.skeleton.device, dtype=torch.float32)
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def constrain_joint(self, joint):
        if joint.min_limit is not None:
            joint.position = torch.max(joint.position, joint.min_limit)
        if joint.max_limit is not None:
            joint.position = torch.min(joint.position, joint.max_limit)

    def solve(self):
        iteration = 0
        while torch.norm(self.skeleton.joints[-1].position - self.target) > self.tolerance and iteration < self.max_iterations:
            iteration += 1

            # Backward Reaching
            self.skeleton.joints[-1].position = self.target
            for i in range(len(self.skeleton.joints) - 2, -1, -1):
                direction = self.skeleton.joints[i].position - self.skeleton.joints[i + 1].position
                direction = direction / torch.norm(direction) * self.skeleton.distances[i]
                self.skeleton.joints[i].position = self.skeleton.joints[i + 1].position + direction
                self.constrain_joint(self.skeleton.joints[i])

            # Forward Reaching
            for i in range(len(self.skeleton.joints) - 1):
                direction = self.skeleton.joints[i + 1].position - self.skeleton.joints[i].position
                direction = direction / torch.norm(direction) * self.skeleton.distances[i]
                self.skeleton.joints[i + 1].position = self.skeleton.joints[i].position + direction
                self.constrain_joint(self.skeleton.joints[i + 1])

        return self.skeleton


if __name__ == '__main__':
    device = 'cpu'
    joints = [
        Joint([0, 0, 0], device=device),
        Joint([1, 0, 0], device=device),
        Joint([2, 0, 0], device=device),
        Joint([3, 0, 0], device=device),
        Joint([4, 0, 0], device=device),
        Joint([5, 0, 0], device=device),
        Joint([6, 0, 0], device=device),
        Joint([7, 0, 0], device=device),
        Joint([8, 0, 0], device=device),
        Joint([9, 0, 0], device=device),
        Joint([10, 0, 0], device=device)
    ]
    skeleton = Skeleton(joints)
    fabrik = FABRIK(skeleton, target=[12, 0, 0])
    solved_skeleton = fabrik.solve()
    print("Solved Skeleton:")
    for joint in solved_skeleton.joints:
        print(joint)
